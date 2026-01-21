import "dotenv/config";
import express from "express";
import { WebSocketServer } from "ws";
import {
  ElevenLabsClient,
  RealtimeEvents,
  AudioFormat,
} from "@elevenlabs/elevenlabs-js";

const app = express();
const server = app.listen(3001, () => {
  console.log("Server listening on http://localhost:3001");
});

const wss = new WebSocketServer({ server });

const elevenlabs = new ElevenLabsClient({
  apiKey: process.env.ELEVENLABS_API_KEY,
});

async function callBedrockAgent(text, sessionId) {
  const response = await fetch(process.env.BEDROCK_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      inputText: text,
      sessionId, // MUY IMPORTANTE para mantener contexto
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Bedrock error: ${errText}`);
  }

  const data = await response.json();
  return data;
}


async function streamTextToSpeech(text, clientWs) {
  console.log("🔊 TTS start:", text);
  
  try {
    const audioStream = await elevenlabs.textToSpeech.stream(
      "JBFqnCBsd6RMkjVDRZzb", // voiceId
      {
        modelId: "eleven_multilingual_v2",
        text,
        outputFormat: "mp3_44100_128",
        voiceSettings: {
          stability: 0,
          similarityBoost: 1.0,
          useSpeakerBoost: true,
          speed: 1.0,
        },
      }
    );

    // Opción 1: Acumular y enviar completo (mejor para chunks pequeños)
    const chunks = [];
    for await (const chunk of audioStream) {
      chunks.push(chunk);
    }
    
    // Concatenar todos los chunks en un solo buffer
    const completeAudio = Buffer.concat(chunks);
    console.log(`✅ TTS complete - ${completeAudio.length} bytes total`);
    
    // Enviar el audio completo
    clientWs.send(completeAudio);
    
    clientWs.send(
      JSON.stringify({
        type: "tts_audio_end",
      })
    );
    
  } catch (err) {
    console.error("❌ TTS error:", err);
    clientWs.send(
      JSON.stringify({
        type: "error",
        error: "TTS failed",
      })
    );
  }
}

// Opción 2: Streaming con chunks más grandes (mejor para latencia baja)
async function streamTextToSpeechChunked(text, clientWs) {
  console.log("🔊 TTS start:", text);
  
  try {
    const audioStream = await elevenlabs.textToSpeech.stream(
      "JBFqnCBsd6RMkjVDRZzb",
      {
        modelId: "eleven_multilingual_v2",
        text,
        outputFormat: "mp3_44100_128",
        voiceSettings: {
          stability: 0,
          similarityBoost: 1.0,
          useSpeakerBoost: true,
          speed: 1.0,
        },
      }
    );

    clientWs.send(
      JSON.stringify({
        type: "tts_audio_start"
      })
    );

    let chunkCount = 0;
    let buffer = [];
    const MIN_CHUNK_SIZE = 8192; // Acumular al menos 8KB antes de enviar

    for await (const chunk of audioStream) {
      buffer.push(chunk);
      const bufferSize = buffer.reduce((acc, c) => acc + c.length, 0);
      
      // Enviar cuando tengamos suficiente data
      if (bufferSize >= MIN_CHUNK_SIZE) {
        const combined = Buffer.concat(buffer);
        chunkCount++;
        console.log(`📦 Sending TTS chunk #${chunkCount} - ${combined.length} bytes`);
        clientWs.send(combined);
        buffer = [];
      }
    }

    // Enviar lo que queda en el buffer
    if (buffer.length > 0) {
      const combined = Buffer.concat(buffer);
      chunkCount++;
      console.log(`📦 Sending final TTS chunk #${chunkCount} - ${combined.length} bytes`);
      clientWs.send(combined);
    }

    console.log(`✅ TTS finished - ${chunkCount} total chunks sent`);

    clientWs.send(
      JSON.stringify({
        type: "tts_audio_end",
      })
    );
    
  } catch (err) {
    console.error("❌ TTS error:", err);
    clientWs.send(
      JSON.stringify({
        type: "error",
        error: "TTS failed",
      })
    );
  }
}

wss.on("connection", async (clientWs) => {
  console.log("✅ Client connected");
  let elevenConnected = true;

  // 1️⃣ Abrimos conexión realtime con ElevenLabs
  const connection = await elevenlabs.speechToText.realtime.connect({
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

  // 2️⃣ Eventos desde ElevenLabs → frontend
  connection.on(RealtimeEvents.SESSION_STARTED, (data) => {
    console.log("🎙️ ElevenLabs session started", data);
  });

  connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, (transcript) => {
    clientWs.send(
      JSON.stringify({
        type: "partial",
        data: transcript,
      })
    );
  });

  let sessionId = null;
  connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, async (transcript) => {
    clientWs.send(
      JSON.stringify({
        type: "final",
        data: transcript,
      })
    );
    
    const text = transcript.text?.trim();
    console.log("📝 FINAL TEXT:", text);

    if (!text) return;

    try {
      clientWs.send(JSON.stringify({
        type: "thinking"
      }));

      // 1️⃣ Enviar texto a Bedrock
      const bedrockResponse = await callBedrockAgent(text, sessionId);

      sessionId = bedrockResponse.sessionId;
      const agentText = bedrockResponse.response;

      clientWs.send(JSON.stringify({
        type: "agent_text",
        text: agentText,
      }));

      console.log("🤖 Bedrock:", agentText);

      // 2️⃣ Enviar respuesta por TTS
      await streamTextToSpeech(agentText, clientWs);

    } catch (err) {
      console.error("❌ Bedrock/TTS error", err);
      clientWs.send(JSON.stringify({
        type: "error",
        error: "Agent failed",
      }));
    }
  });

  connection.on(
    RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
    (transcript) => {
      clientWs.send(
        JSON.stringify({
          type: "final_with_timestamps",
          data: transcript,
        })
      );
    }
  );

  connection.on(RealtimeEvents.ERROR, (error) => {
    console.error("❌ ElevenLabs error", error);
    clientWs.send(
      JSON.stringify({
        type: "error",
        error,
      })
    );
  });

  connection.on(RealtimeEvents.CLOSE, () => {
    console.log("🔌 ElevenLabs connection closed");
    elevenConnected = false;
  });

  // 3️⃣ Audio desde frontend → ElevenLabs
  clientWs.on("message", (message) => {
    // Mensajes de control (JSON)
    if (typeof message === "string") {
      try {
        const data = JSON.parse(message);

        if (data.event === "stop" && elevenConnected) {
          console.log("⏹️ Commit requested");
          connection.commit();
        }
      } catch (err) {
        console.error("❌ Error parsing message:", err);
      }

      return;
    }

    // Audio chunk (Buffer PCM16)
    if (!elevenConnected) {
      return;
    }

    const audioBuffer = Buffer.from(message);
    const audioBase64 = audioBuffer.toString("base64");

    connection.send({
      audioBase64,
      sampleRate: 16000,
    });
  });

  clientWs.on("close", () => {
    console.log("❌ Client disconnected");
    if (elevenConnected) {
      connection.close();
    }
  });
});