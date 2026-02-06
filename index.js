import "dotenv/config";
import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";
import {
  ElevenLabsClient,
  RealtimeEvents,
  AudioFormat,
} from "@elevenlabs/elevenlabs-js";
import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";

const app = express();

app.use(cors({
  origin: [
    'https://elevenlabs-front-six.vercel.app',
    'http://localhost:5173',
    'http://localhost:3000'
  ],
  methods: ['GET', 'POST', 'OPTIONS'],
  credentials: true
}));

// 🆕 Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

const server = app.listen(3001, () => {
  console.log("Server listening on http://localhost:3001");
});

const wss = new WebSocketServer({ server });

const elevenlabs = new ElevenLabsClient({
  apiKey: process.env.ELEVENLABS_API_KEY,
});

const bedrockClient = new BedrockAgentRuntimeClient({
  region: process.env.AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
});

async function processUserIntent(text, clientWs, sessionIdState) {
  if (!text) return;
  
  try {
    clientWs.send(JSON.stringify({ type: "thinking" }));

    let pendingText = "";
    let ttsQueue = Promise.resolve();
    let currentSessionId = sessionIdState;

    // Llamada a Bedrock (mismo flujo que ya tenías)
      for await (const event of streamBedrockAgent(text, currentSessionId)) {
        if (event.type === "chunk") {
          pendingText += event.text;

          const { sentences, remaining } = extractCompleteSentences(pendingText);
          
          if (sentences) {
            const cleaned = cleanTextForTTS(sentences);
            if (cleaned) {
              // 🆕 Solo enviamos al frontend cuando tenemos una oración limpia y completa
              clientWs.send(JSON.stringify({
                type: "agent_text_chunk", // Reutilizamos el tipo pero con texto limpio
                accumulated: cleaned 
              }));
              
              ttsQueue = ttsQueue.then(() => streamTextToSpeechPCM(cleaned, clientWs));
            }
            pendingText = remaining;
          }
        }
      
      if (event.type === "complete") {
        if (pendingText.trim()) {
          const cleanedPending = cleanTextForTTS(pendingText);
          ttsQueue = ttsQueue.then(() => streamTextToSpeechPCM(cleanedPending, clientWs));
        }
        await ttsQueue;
        clientWs.send(JSON.stringify({ 
          type: "agent_complete", 
          text: cleanTextForTTS(event.text) 
        }));
        return event.sessionId; // Retornamos el ID para mantener el hilo
      }
    }
  } catch (err) {
    console.error("❌ Error en procesamiento:", err);
    clientWs.send(JSON.stringify({ type: "error", error: "Failed to process request" }));
  }
}

// 🆕 Función auxiliar para esperar conexión con retry
async function connectWithRetry(connectFn, maxRetries = 5, delayMs = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      console.log(`🔄 Attempting ElevenLabs connection (attempt ${i + 1}/${maxRetries})...`);
      const connection = await connectFn();
      console.log("✅ ElevenLabs connected successfully");
      return connection;
    } catch (err) {
      console.error(`❌ Connection attempt ${i + 1} failed:`, err.message);
      
      if (i < maxRetries - 1) {
        console.log(`⏳ Retrying in ${delayMs}ms...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      } else {
        throw new Error(`Failed to connect after ${maxRetries} attempts: ${err.message}`);
      }
    }
  }
}

async function* streamBedrockAgent(text, sessionId) {
  const finalSessionId = sessionId || `session-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  
  const command = new InvokeAgentCommand({
    agentId: process.env.BEDROCK_AGENT_ID,
    agentAliasId: process.env.BEDROCK_AGENT_ALIAS_ID,
    sessionId: finalSessionId,
    inputText: text,
    enableTrace: false,
    streamingConfigurations: {
      streamFinalResponse: true,
    },
  });

  try {
    const response = await bedrockClient.send(command);
    let accumulatedText = "";

    for await (const event of response.completion) {
      if (event.chunk?.bytes) {
        const decodedChunk = new TextDecoder().decode(event.chunk.bytes);
        accumulatedText += decodedChunk;
        
        yield {
          type: "chunk",
          text: decodedChunk,
          accumulated: accumulatedText,
        };
      }
    }

    yield {
      type: "complete",
      text: accumulatedText,
      sessionId: finalSessionId,
    };
  } catch (error) {
    console.error("❌ Bedrock error:", error);
    throw error;
  }
}

function cleanTextForTTS(text) {
  const cleanedText = text
    .replace(/\\n+/g, ' ')
    .replace(/\n+/g, ' ')
    .replace(/\t+/g, ' ')
    .replace(/\r/g, '')
    .replace(/<[^>]+>.*?<\/[^>]+>/gs, '')
    .replace(/<[^>]+>/g, '')
    .replace(/\s{2,}/g, ' ')
    .replace(/[*_~`]/g, '')
    .trim();
  return cleanedText;
}

async function streamTextToSpeechPCM(text, clientWs) {
  try {
    const audioStream = await elevenlabs.textToSpeech.stream(
      "gKg8M8yuhJHaZpgdtyrn",
      {
        modelId: "eleven_turbo_v2_5",
        languageCode: "es",
        text,
        outputFormat: "pcm_24000",
        voiceSettings: {
          stability: 0.5,
          similarityBoost: 0.8,
          useSpeakerBoost: true,
          speed: 1.05,
        },
      }
    );

    clientWs.send(
      JSON.stringify({
        type: "tts_audio_start",
        format: "pcm",
        sampleRate: 24000,
        channels: 1,
        bitDepth: 16,
      })
    );

    let buffer = [];
    let leftoverByte = null;
    
    const MIN_CHUNK_SIZE = 4096;
    const MAX_BUFFER_SIZE = 12288;
    const CHUNK_TIMEOUT = 50;

    let lastSendTime = Date.now();

    for await (const chunk of audioStream) {
      if (!chunk || chunk.length === 0) {
        console.warn("⚠️ ElevenLabs sent empty chunk, skipping");
        continue;
      }

      buffer.push(chunk);
      const bufferSize = buffer.reduce((acc, c) => acc + c.length, 0);
      const timeSinceLastSend = Date.now() - lastSendTime;
      
      const shouldSend = bufferSize >= MIN_CHUNK_SIZE || 
                        bufferSize >= MAX_BUFFER_SIZE ||
                        (bufferSize > 0 && timeSinceLastSend > CHUNK_TIMEOUT);
      
      if (shouldSend) {
        let combined = Buffer.concat(buffer);
        
        if (leftoverByte !== null) {
          combined = Buffer.concat([Buffer.from([leftoverByte]), combined]);
          leftoverByte = null;
        }
        
        if (combined.length % 2 !== 0) {
          leftoverByte = combined[combined.length - 1];
          combined = combined.slice(0, -1);
        }
        
        if (combined.length > 0) {
          clientWs.send(combined);
          lastSendTime = Date.now();
        }
        
        buffer = [];
      }
    }

    if (buffer.length > 0 || leftoverByte !== null) {
      let combined = buffer.length > 0 ? Buffer.concat(buffer) : Buffer.alloc(0);
      
      if (leftoverByte !== null) {
        combined = Buffer.concat([Buffer.from([leftoverByte]), combined]);
        leftoverByte = null;
      }
      
      if (combined.length % 2 !== 0) {
        combined = combined.slice(0, -1);
      }
      
      if (combined.length > 0) {
        clientWs.send(combined);
      }
    }

    clientWs.send(JSON.stringify({ type: "tts_audio_end" }));
    
  } catch (err) {
    console.error("❌ TTS error:", err);
    clientWs.send(JSON.stringify({ type: "error", error: "TTS failed" }));
  }
}

function extractCompleteSentences(text) {
  const MIN_LENGTH = 8;
  const matches = [...text.matchAll(/[.!?:]+(?:\s+|$)|,(?=\s+[A-Z])/g)];
  
  if (matches.length === 0) {
    if (text.length > 100) {
      const lastComma = Math.max(text.lastIndexOf(','), text.lastIndexOf(';'), text.lastIndexOf(':'));
      if (lastComma > 60) {
        return {
          sentences: text.substring(0, lastComma + 1).trim(),
          remaining: text.substring(lastComma + 1).trim()
        };
      }
    }
    return { sentences: "", remaining: text };
  }
  
  const lastMatch = matches[matches.length - 1];
  const lastIndex = lastMatch.index + lastMatch[0].length;
  const sentences = text.substring(0, lastIndex).trim();
  const remaining = text.substring(lastIndex).trim();
  
  if (sentences.length < MIN_LENGTH ) {
    return { sentences: "", remaining: text };
  }
  
  return { sentences, remaining };
}

wss.on("connection", async (clientWs) => {
    console.log("✅ Client connected");
    let connection = null;
    let isConnected = false;
    let keepAliveInterval = null;
    let lastAudioTime = Date.now();
    let sessionId = null;

    clientWs.send(JSON.stringify({ type: "initializing", message: "Connecting to speech service..." }));

    try {
        connection = await connectWithRetry(async () => {
            return await elevenlabs.speechToText.realtime.connect({
                modelId: "scribe_v2_realtime",
                audioFormat: AudioFormat.PCM_16000,
                sampleRate: 16000,
                languageCode: "es",
                commitStrategy: "vad",
                vadThreshold: 0.75,
                vadSilenceThresholdSecs: 1.5,
                minSpeechDurationMs: 150,
                minSilenceDurationMs: 600,
            });
        });

        isConnected = true;

        connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, (transcript) => {
            clientWs.send(JSON.stringify({ type: "partial", data: transcript }));
        });

        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, async (transcript) => {
            clientWs.send(JSON.stringify({ type: "final", data: transcript }));
            sessionId = await processUserIntent(transcript.text?.trim(), clientWs, sessionId);
        });

        connection.on(RealtimeEvents.ERROR, (err) => console.error("❌ ElevenLabs Error:", err));
        
        connection.on(RealtimeEvents.CLOSE, () => {
            isConnected = false;
            if (keepAliveInterval) clearInterval(keepAliveInterval);
        });

        clientWs.send(JSON.stringify({ type: "ready", message: "Service ready" }));

        keepAliveInterval = setInterval(() => {
            if (isConnected && connection && (Date.now() - lastAudioTime > 2000)) {
                try {
                    const silence = Buffer.alloc(1024, 0);
                    connection.send({ audioBase64: silence.toString("base64"), sampleRate: 16000 });
                } catch (e) { console.error("Keep-alive error", e); }
            }
        }, 2000);

        clientWs.on("message", async (message) => {
            if (typeof message === "string" || (Buffer.isBuffer(message) && message.length < 500)) {
                try {
                    const data = JSON.parse(message.toString());
                      // index.js -> dentro de clientWs.on("message")
                      if (data.type === "text_input") {
                          console.log("✍️ Nueva entrada de texto (Interrupción):", data.text);
                          
                          // Al llamar a processUserIntent, el await asegura que se procese, 
                          // pero el sessionId actualizado permite que Bedrock sepa que es una continuación.
                          sessionId = await processUserIntent(data.text, clientWs, sessionId);
                          return;
                      }
                    if (data.event === "stop" && isConnected) connection.commit();
                } catch (e) { /* Not JSON */ }
            } else if (isConnected && connection) {
                lastAudioTime = Date.now();
                connection.send({ audioBase64: Buffer.from(message).toString("base64"), sampleRate: 16000 });
            }
        });

        clientWs.on("close", () => {
            if (keepAliveInterval) clearInterval(keepAliveInterval);
            if (connection) connection.close();
            isConnected = false;
        });

    } catch (err) {
        console.error("❌ Connection failed:", err);
        if (clientWs.readyState === 1) clientWs.send(JSON.stringify({ type: "error", error: "Service Unavailable" }));
    }
});