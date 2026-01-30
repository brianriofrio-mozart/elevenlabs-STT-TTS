import "dotenv/config";
import express from "express";
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

// 🆕 FUNCIÓN MEJORADA CON ALINEACIÓN CORRECTA
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
    let leftoverByte = null; // 🆕 Para manejar bytes impares
    
    // 🆕 Configuración optimizada
    const MIN_CHUNK_SIZE = 4096;  // Mínimo para enviar (múltiplo de 2)
    const MAX_BUFFER_SIZE = 12288; // Máximo antes de forzar envío
    const CHUNK_TIMEOUT = 50; // ms máximo de espera

    let lastSendTime = Date.now();

    for await (const chunk of audioStream) {
      // 🆕 Validar que el chunk no esté vacío
      if (!chunk || chunk.length === 0) {
        console.warn("⚠️ ElevenLabs sent empty chunk, skipping");
        continue;
      }

      buffer.push(chunk);
      const bufferSize = buffer.reduce((acc, c) => acc + c.length, 0);
      const timeSinceLastSend = Date.now() - lastSendTime;
      
      // 🆕 Enviar si cumple condiciones: tamaño O timeout
      const shouldSend = bufferSize >= MIN_CHUNK_SIZE || 
                        bufferSize >= MAX_BUFFER_SIZE ||
                        (bufferSize > 0 && timeSinceLastSend > CHUNK_TIMEOUT);
      
      if (shouldSend) {
        let combined = Buffer.concat(buffer);
        
        // 🆕 Combinar con leftover byte si existe
        if (leftoverByte !== null) {
          combined = Buffer.concat([Buffer.from([leftoverByte]), combined]);
          leftoverByte = null;
        }
        
        // 🆕 CRÍTICO: Asegurar alineación a 16-bit (par)
        if (combined.length % 2 !== 0) {
          // Guardar último byte para el siguiente chunk
          leftoverByte = combined[combined.length - 1];
          combined = combined.slice(0, -1);
          
          console.log(`🔧 Aligned chunk: ${combined.length + 1} → ${combined.length} bytes (saved 1 byte)`);
        }
        
        // Solo enviar si hay datos después de alinear
        if (combined.length > 0) {
          clientWs.send(combined);
          lastSendTime = Date.now();
        }
        
        buffer = [];
      }
    }

    // 🆕 Procesar buffer final
    if (buffer.length > 0 || leftoverByte !== null) {
      let combined = buffer.length > 0 ? Buffer.concat(buffer) : Buffer.alloc(0);
      
      if (leftoverByte !== null) {
        combined = Buffer.concat([Buffer.from([leftoverByte]), combined]);
        leftoverByte = null;
      }
      
      // Alinear chunk final
      if (combined.length % 2 !== 0) {
        console.warn(`⚠️ Final chunk unaligned (${combined.length}B), truncating last byte`);
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

  try {
    connection = await elevenlabs.speechToText.realtime.connect({
      modelId: "scribe_v2_realtime",
      audioFormat: AudioFormat.PCM_16000,
      sampleRate: 16000,
      languageCode: "es",
      includeTimestamps: true,
      commitStrategy: "vad",
      vadThreshold: 0.75,
      vadSilenceThresholdSecs: 0.4,
      minSpeechDurationMs: 150,
      minSilenceDurationMs: 300,
    });

    isConnected = true;

    keepAliveInterval = setInterval(() => {
      if (!isConnected || !connection) return;

      const timeSinceLastAudio = Date.now() - lastAudioTime;
      
      if (timeSinceLastAudio > 2000) {
        try {
          const silenceBuffer = Buffer.alloc(1024, 0);
          connection.send({
            audioBase64: silenceBuffer.toString("base64"),
            sampleRate: 16000,
          });
        } catch (err) {
          console.error("❌ Keep-alive error:", err);
        }
      }
    }, 2000);

    connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, (transcript) => {
      clientWs.send(JSON.stringify({ type: "partial", data: transcript }));
    });

    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, async (transcript) => {
      clientWs.send(JSON.stringify({ type: "final", data: transcript }));
      
      const text = transcript.text?.trim();
      if (!text) return;

      try {
        clientWs.send(JSON.stringify({ type: "thinking" }));

        let pendingText = "";
        let ttsQueue = Promise.resolve();

        for await (const event of streamBedrockAgent(text, sessionId)) {
          if (event.type === "chunk") {
            pendingText += event.text;

            clientWs.send(JSON.stringify({
              type: "agent_text_chunk",
              chunk: event.text,
              accumulated: event.accumulated,
            }));

            const { sentences, remaining } = extractCompleteSentences(pendingText);
            
            if (sentences) {
              ttsQueue = ttsQueue.then(() => 
                streamTextToSpeechPCM(sentences, clientWs)
              );
              pendingText = remaining;
            }
          }
          
          if (event.type === "complete") {
            sessionId = event.sessionId;
            
            if (pendingText.trim()) {
              ttsQueue = ttsQueue.then(() =>
                streamTextToSpeechPCM(pendingText, clientWs)
              );
            }

            await ttsQueue;

            clientWs.send(JSON.stringify({
              type: "agent_complete",
              text: event.text,
            }));
          }
        }
      } catch (err) {
        console.error("❌ Error:", err);
        clientWs.send(JSON.stringify({
          type: "error",
          error: err.message || "Agent failed",
        }));
      }
    });

    connection.on(RealtimeEvents.ERROR, (error) => {
      console.error("❌ ElevenLabs error:", error);
      clientWs.send(JSON.stringify({ type: "error", error }));
    });

    connection.on(RealtimeEvents.CLOSE, () => {
      console.log("🔌 ElevenLabs connection closed");
      isConnected = false;
      
      if (keepAliveInterval) {
        clearInterval(keepAliveInterval);
        keepAliveInterval = null;
      }
    });

    clientWs.on("message", (message) => {
      if (typeof message === "string") {
        try {
          const data = JSON.parse(message);
          if (data.event === "stop" && isConnected && connection) {
            connection.commit();
          }
        } catch (err) {
          console.error("❌ Error parsing message:", err);
        }
        return;
      }

      if (!isConnected || !connection) return;

      lastAudioTime = Date.now();

      try {
        const audioBuffer = Buffer.from(message);
        connection.send({
          audioBase64: audioBuffer.toString("base64"),
          sampleRate: 16000,
        });
      } catch (err) {
        console.error("❌ Error sending audio:", err);
        isConnected = false;
      }
    });

    clientWs.on("close", () => {
      console.log("❌ Client disconnected");

      if (keepAliveInterval) {
        clearInterval(keepAliveInterval);
        keepAliveInterval = null;
      }

      if (isConnected && connection) {
        try {
          connection.close();
        } catch (err) {
          console.error("Error closing connection:", err.message);
        }
      }

      isConnected = false;
    });

  } catch (err) {
    console.error("❌ Failed to establish connection:", err);
    isConnected = false;

    if (keepAliveInterval) {
      clearInterval(keepAliveInterval);
    }
    
    if (clientWs.readyState === clientWs.OPEN) {
      clientWs.send(JSON.stringify({
        type: "error",
        error: "Failed to connect to speech service",
      }));
      clientWs.close();
    }
  }
});