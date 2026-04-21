import crypto from "node:crypto";
import { GoogleGenerativeAI } from "@google/generative-ai";
import Groq from "groq-sdk";
import env from "../config/env.js";
import { query as dbQuery } from "../db/neon.js";
import { embedText } from "./embedding.service.js";
import {
  readPositiveInt,
  parseBoolean,
  isPlainObject,
  normalizeMetadata,
} from "../utils/helpers.js";

let ensureSchemaPromise = null;
let vectorSearchEnabled = false;
let vectorSearchWarning = "";
const vectorColumnType = env.embeddingDimensions > 2000 ? "halfvec" : "vector";
const vectorIndexOperatorClass =
  vectorColumnType === "halfvec" ? "halfvec_cosine_ops" : "vector_cosine_ops";
const vectorIndexMethod = vectorColumnType === "halfvec" ? "hnsw" : "ivfflat";
const SOURCE_TYPE_DOCUMENT = "documento";
const SOURCE_TYPE_DATABASE = "base_datos";

function normalizeSourceType(value, fallback = SOURCE_TYPE_DOCUMENT) {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().toLowerCase().replace(/\s+/g, "_");

  if (normalized === "base_de_datos" || normalized === "database" || normalized === "db") {
    return SOURCE_TYPE_DATABASE;
  }

  if (normalized === "document") {
    return SOURCE_TYPE_DOCUMENT;
  }

  if (normalized === SOURCE_TYPE_DOCUMENT || normalized === SOURCE_TYPE_DATABASE) {
    return normalized;
  }

  return fallback;
}

function normalizeIdentifierValue(value, fallback) {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }

  return fallback;
}

function parseMetadataRow(value) {
  if (isPlainObject(value)) {
    return value;
  }

  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return isPlainObject(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }

  return {};
}

function sanitizeIdentifier(identifier, options = {}) {
  const { allowQualified = false, label = "identifier" } = options;

  if (typeof identifier !== "string" || !identifier.trim()) {
    throw new Error(`El identificador '${label}' no es valido.`);
  }

  const normalized = identifier.trim();
  const parts = allowQualified ? normalized.split(".") : [normalized];
  const validPart = /^[A-Za-z_][A-Za-z0-9_]*$/;

  if (!parts.every((part) => validPart.test(part))) {
    throw new Error(`El identificador '${label}' no es seguro para SQL.`);
  }

  return parts.join(".");
}

function normalizeEmbeddingValues(values) {
  if (!Array.isArray(values)) {
    return [];
  }

  return values.map((value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  });
}

function toVectorLiteral(values = []) {
  return `[${values.join(",")}]`;
}

function buildDefaultDocumentIdentifier(fuente, contenido, metadata = {}) {
  const fileName = typeof metadata.archivo === "string" ? metadata.archivo.trim() : "";
  const fragment = Number.isInteger(metadata.fragmento) ? metadata.fragmento : null;

  if (fileName && fragment) {
    return `${fileName}:fragmento-${fragment}`;
  }

  if (fileName) {
    return fileName;
  }

  const hash = crypto
    .createHash("sha1")
    .update(`${fuente}|${contenido}`)
    .digest("hex")
    .slice(0, 16);

  return `doc:${hash}`;
}

function enrichDocumentMetadata(metadata, fuente, contenido) {
  const normalized = normalizeMetadata(metadata);
  const tipoFuente = normalizeSourceType(normalized.tipoFuente, SOURCE_TYPE_DOCUMENT);
  const identificador = normalizeIdentifierValue(
    normalized.identificador,
    buildDefaultDocumentIdentifier(fuente, contenido, normalized)
  );

  return {
    ...normalized,
    tipoFuente,
    identificador,
  };
}

function parseNumericScore(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function buildDocumentContextItem(row, fallbackScore = 0) {
  const metadata = parseMetadataRow(row.metadata);
  const tipoFuente = normalizeSourceType(metadata.tipoFuente, SOURCE_TYPE_DOCUMENT);
  const identificador = normalizeIdentifierValue(metadata.identificador, `doc:${row.id ?? "sin-id"}`);

  return {
    id: row.id,
    fuente: row.fuente,
    contenido: row.contenido,
    metadata,
    tipoFuente,
    identificador,
    score: parseNumericScore(row.score, fallbackScore),
  };
}

function buildRecordContextItem(row, safeTableName) {
  const metadata = parseMetadataRow(row.metadata);
  const recordId = row.id ?? "sin-id";
  const identificador = normalizeIdentifierValue(metadata.identificador, `${safeTableName}:${recordId}`);

  return {
    id: recordId,
    fuente: row.fuente || env.ragRecordsSourceLabel || safeTableName,
    contenido: row.contenido,
    metadata,
    tipoFuente: SOURCE_TYPE_DATABASE,
    identificador,
    score: parseNumericScore(row.score),
  };
}

function buildPromptContext(contextItems = []) {
  return contextItems
    .map(
      (item, index) =>
        `Contexto ${index + 1} [${item.tipoFuente}] (${item.fuente} | ${item.identificador}):\n${item.contenido}`
    )
    .join("\n\n");
}

function normalizeAnswerText(value) {
  return String(value || "")
    .replace(/\r/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function stripListPrefix(line) {
  return line.trim().replace(/^([*-]\s+|\d+\.\s+)/, "");
}

function isInternalAnswerLine(line) {
  const normalized = stripListPrefix(line).toLowerCase();

  if (!normalized) {
    return false;
  }

  return [
    /^question:/i,
    /^pregunta:/i,
    /^constraint\s*\d+:/i,
    /^restriccion\s*\d+:/i,
    /^context\s*\d+:/i,
    /^contexto\s*\d+:/i,
    /^draft:/i,
    /^borrador:/i,
    /^only based on context\?/i,
    /^insufficient evidence\?/i,
    /^cite sources\?/i,
    /^cites sources\?/i,
    /^only final answer\?/i,
    /^no internal blocks\?/i,
    /^spanish\?/i,
    /^solo con base en el contexto\?/i,
    /^evidencia insuficiente\?/i,
    /^cita fuentes\?/i,
    /^solo respuesta final\?/i,
    /^sin bloques internos\?/i,
    /^espanol\?/i,
  ].some((pattern) => pattern.test(normalized));
}

function removeChecklistFragments(text) {
  return String(text || "")
    .replace(
      /\b(?:cite sources\?|cites sources\?|only final answer\?|no internal blocks\?|only based on context\?|insufficient evidence\?|spanish\?)\s*yes\.?/gi,
      ""
    )
    .replace(
      /\b(?:cita fuentes\?|solo respuesta final\?|sin bloques internos\?|solo con base en el contexto\?|evidencia insuficiente\?|espanol\?)\s*s[ií]\.?/gi,
      ""
    )
    .replace(/\s+/g, " ")
    .trim();
}

function pickQuotedAnswerCandidate(text) {
  const raw = String(text || "");
  const matches = [...raw.matchAll(/"([^"\n]{40,})"/g)].map((item) => item[1].trim());

  if (!matches.length) {
    return "";
  }

  const candidates = matches.filter(
    (chunk) => !/(question:|pregunta:|constraint\s*\d+:|context\s*\d+:|contexto\s*\d+:|draft:|borrador:)/i.test(chunk)
  );

  if (!candidates.length) {
    return "";
  }

  const score = (chunk) => {
    let points = 0;

    if (/\b(el|la|los|las|de|del|para|con|se|que|y|en)\b/i.test(chunk)) {
      points += 2;
    }

    if (/[áéíóúñ]/i.test(chunk)) {
      points += 1;
    }

    if (/\|\s*doc:\w+/i.test(chunk) || /\(.*\|\s*doc:/i.test(chunk)) {
      points += 1;
    }

    return points;
  };

  return candidates
    .sort((a, b) => {
      const byScore = score(b) - score(a);
      if (byScore !== 0) return byScore;
      return b.length - a.length;
    })
    .at(0);
}

function dedupeSentences(text) {
  const normalized = text
    .replace(/\)\s*([A-Za-z])/g, ") $1")
    .replace(/([a-z0-9áéíóúñ][.!?])([A-ZÁÉÍÓÚÑ])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim();

  if (!normalized) {
    return "";
  }

  const chunks = normalized.split(/(?<=[.!?])\s+/);
  const unique = [];
  const seen = new Set();

  for (const chunk of chunks) {
    const sentence = chunk.trim();

    if (!sentence) {
      continue;
    }

    const key = sentence.toLowerCase().replace(/\s+/g, " ").trim();

    if (seen.has(key) && key.length > 20) {
      continue;
    }

    seen.add(key);
    unique.push(sentence);
  }

  return unique.join(" ").trim();
}

function sanitizeModelAnswer(value) {
  const raw = normalizeAnswerText(value);

  if (!raw) {
    return "";
  }

  const filteredLines = raw.split("\n").filter((line) => !isInternalAnswerLine(line));
  let cleaned = normalizeAnswerText(filteredLines.join("\n"));

  if (!cleaned) {
    return raw;
  }

  const allListLines = cleaned
    .split("\n")
    .every((line) => !line.trim() || /^([*-]\s+|\d+\.\s+)/.test(line.trim()));

  if (allListLines) {
    cleaned = normalizeAnswerText(
      cleaned
        .split("\n")
        .map((line) => stripListPrefix(line))
        .join(" ")
    );
  }

  const hasInternalScaffold =
    /(?:question:|pregunta:|constraint\s*\d+:|context\s*\d+:|contexto\s*\d+:|draft:|borrador:|only based on context\?|insufficient evidence\?|cite sources\?|cites sources\?|only final answer\?|no internal blocks\?|spanish\?|solo con base en el contexto\?|evidencia insuficiente\?|cita fuentes\?|solo respuesta final\?|sin bloques internos\?|espanol\?)/i.test(
      raw
    );

  cleaned = removeChecklistFragments(cleaned);

  if (hasInternalScaffold) {
    const quotedCandidate = pickQuotedAnswerCandidate(cleaned || raw);

    if (quotedCandidate) {
      cleaned = quotedCandidate;
    }
  }

  if (hasInternalScaffold) {
    const paragraphs = cleaned
      .split(/\n{2,}/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean);

    if (paragraphs.length > 1) {
      cleaned = paragraphs[paragraphs.length - 1];
    }
  }

  cleaned = dedupeSentences(cleaned);
  return cleaned || raw;
}

async function ensureDocumentosRagSchema() {
  if (ensureSchemaPromise) {
    return ensureSchemaPromise;
  }

  ensureSchemaPromise = (async () => {
    await dbQuery(`
      ALTER TABLE documentos_rag
      ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    `);

    await dbQuery(`
      CREATE INDEX IF NOT EXISTS idx_documentos_rag_metadata_gin
      ON documentos_rag USING GIN (metadata)
    `);

    vectorSearchEnabled = false;
    vectorSearchWarning = "";

    try {
      await dbQuery("CREATE EXTENSION IF NOT EXISTS vector");

      const vectorColumnInfo = await dbQuery(
        `
          SELECT atttypmod, atttypid::regtype::text AS data_type
          FROM pg_attribute
          WHERE attrelid = 'documentos_rag'::regclass
            AND attname = 'embedding_vector'
            AND NOT attisdropped
          LIMIT 1
        `
      );

      const existingTypmod = Number(vectorColumnInfo.rows?.[0]?.atttypmod);
      const existingDataType = String(vectorColumnInfo.rows?.[0]?.data_type || "").trim();
      const existingDimensions = Number.isInteger(existingTypmod) && existingTypmod > 0
        ? existingTypmod - 4
        : null;

      if (
        (existingDimensions && existingDimensions !== env.embeddingDimensions) ||
        (existingDataType && existingDataType !== vectorColumnType)
      ) {
        await dbQuery("ALTER TABLE documentos_rag DROP COLUMN IF EXISTS embedding_vector");
      }

      await dbQuery(`
        ALTER TABLE documentos_rag
        ADD COLUMN IF NOT EXISTS embedding_vector ${vectorColumnType}(${env.embeddingDimensions})
      `);

      await dbQuery(`
        UPDATE documentos_rag
        SET embedding_vector = embedding::text::${vectorColumnType}
        WHERE embedding_vector IS NULL
          AND embedding IS NOT NULL
          AND embedding::text <> ''
      `);

      vectorSearchEnabled = true;

      try {
        if (vectorIndexMethod === "ivfflat") {
          await dbQuery(`
            CREATE INDEX IF NOT EXISTS idx_documentos_rag_embedding_vector_ivfflat
            ON documentos_rag USING ivfflat (embedding_vector ${vectorIndexOperatorClass})
            WITH (lists = 100)
          `);
        } else {
          await dbQuery(`
            CREATE INDEX IF NOT EXISTS idx_documentos_rag_embedding_vector_hnsw
            ON documentos_rag USING hnsw (embedding_vector ${vectorIndexOperatorClass})
            WITH (m = 16, ef_construction = 64)
          `);
        }
      } catch (indexError) {
        vectorSearchWarning =
          indexError instanceof Error
            ? indexError.message
            : "No se pudo crear indice vectorial; se usara scan vectorial.";

        if (env.nodeEnv === "development") {
          console.warn(`[backend-rag] indice vectorial no disponible: ${vectorSearchWarning}`);
        }
      }
    } catch (vectorError) {
      vectorSearchEnabled = false;
      vectorSearchWarning =
        vectorError instanceof Error ? vectorError.message : String(vectorError || "pgvector no disponible");

      if (env.nodeEnv === "development") {
        console.warn(`[backend-rag] pgvector no disponible, se usara fallback: ${vectorSearchWarning}`);
      }
    }
  })();

  try {
    await ensureSchemaPromise;
  } catch (error) {
    ensureSchemaPromise = null;
    vectorSearchEnabled = false;
    throw error;
  }
}

function buildServiceError(error, fallbackMessage) {
  const message = error instanceof Error ? error.message : String(error || "Error desconocido");
  const normalized = message.toLowerCase();

  let code = "INTERNAL_ERROR";

  if (
    normalized.includes("no puede estar vacia") ||
    normalized.includes("no hay documentos validos") ||
    normalized.includes("debe ser un objeto")
  ) {
    code = "BAD_REQUEST";
  } else if (
    normalized.includes("no esta configurada") ||
    normalized.includes("resource_exhausted") ||
    normalized.includes("429") ||
    normalized.includes("not found") ||
    normalized.includes("fetching from") ||
    normalized.includes("timeout")
  ) {
    code = "DEPENDENCY_ERROR";
  } else if (normalized.includes("relation") && normalized.includes("does not exist")) {
    code = "DATASTORE_ERROR";
  }

  return {
    ok: false,
    code,
    error: env.nodeEnv === "development" ? message : fallbackMessage,
  };
}

function cosineSimilarity(a = [], b = []) {
  if (!a.length || !b.length || a.length !== b.length) {
    return -1;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (!normA || !normB) {
    return -1;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

let _llmModel = null;

function getLlmClient() {
  if (!env.googleApiKey) {
    throw new Error("GOOGLE_API_KEY no esta configurada.");
  }

  if (!_llmModel) {
    _llmModel = new GoogleGenerativeAI(env.googleApiKey).getGenerativeModel({
      model: env.llmModel,
    });
  }

  return _llmModel;
}

let _groqClient = null;

function getGroqClient() {
  if (!env.groqApiKey) {
    throw new Error("GROQ_API_KEY no esta configurada.");
  }

  if (!_groqClient) {
    _groqClient = new Groq({
      apiKey: env.groqApiKey,
    });
  }

  return _groqClient;
}

async function generateWithGroq(prompt) {
  const client = getGroqClient();
  const completion = await client.chat.completions.create({
    model: env.groqModel,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
    temperature: 0.2,
  });

  return completion.choices?.[0]?.message?.content?.trim() || "";
}

async function* generateWithGroqStream(prompt) {
  const client = getGroqClient();
  const stream = await client.chat.completions.create({
    model: env.groqModel,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
    temperature: 0.2,
    stream: true,
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || "";
    if (content) {
      yield content;
    }
  }
}

async function retrieveDocumentContextVector(questionEmbedding, filtrosMetadata, limit) {
  const vectorLiteral = toVectorLiteral(questionEmbedding);
  const queryParams = [vectorLiteral];
  const conditions = ["embedding_vector IS NOT NULL"];

  if (Object.keys(filtrosMetadata).length > 0) {
    queryParams.push(JSON.stringify(filtrosMetadata));
    conditions.push(`metadata @> $${queryParams.length}::jsonb`);
  }

  queryParams.push(limit);

  const dbResult = await dbQuery(
    `
      SELECT
        id,
        fuente,
        contenido,
        metadata,
        (1 - (embedding_vector <=> $1::${vectorColumnType})) AS score
      FROM documentos_rag
      WHERE ${conditions.join(" AND ")}
      ORDER BY embedding_vector <=> $1::${vectorColumnType}
      LIMIT $${queryParams.length}
    `,
    queryParams
  );

  return dbResult.rows.map((row) => buildDocumentContextItem(row));
}

async function retrieveDocumentContextLegacy(questionEmbedding, filtrosMetadata, limit) {
  const queryParams = [];
  const conditions = [];

  if (Object.keys(filtrosMetadata).length > 0) {
    queryParams.push(JSON.stringify(filtrosMetadata));
    conditions.push(`metadata @> $${queryParams.length}::jsonb`);
  }

  const candidateLimit = Math.max(200, limit * 50);
  queryParams.push(candidateLimit);
  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  const dbResult = await dbQuery(
    `
      SELECT id, fuente, contenido, embedding, metadata
      FROM documentos_rag
      ${whereClause}
      ORDER BY id DESC
      LIMIT $${queryParams.length}
    `,
    queryParams
  );

  return dbResult.rows
    .map((row) => {
      const embedding = Array.isArray(row.embedding)
        ? normalizeEmbeddingValues(row.embedding)
        : normalizeEmbeddingValues(JSON.parse(row.embedding || "[]"));

      const score = cosineSimilarity(questionEmbedding, embedding);
      return buildDocumentContextItem({ ...row, score }, score);
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

async function retrieveDocumentContext(questionEmbedding, filtrosMetadata, limit) {
  const safeLimit = readPositiveInt(limit, env.ragTopK);

  if (vectorSearchEnabled) {
    try {
      const items = await retrieveDocumentContextVector(questionEmbedding, filtrosMetadata, safeLimit);
      return {
        items,
        mode: "vector",
        warning: vectorSearchWarning,
      };
    } catch (vectorError) {
      const warning = vectorError instanceof Error ? vectorError.message : String(vectorError || "");
      return {
        items: await retrieveDocumentContextLegacy(questionEmbedding, filtrosMetadata, safeLimit),
        mode: "legacy",
        warning,
      };
    }
  }

  const items = await retrieveDocumentContextLegacy(questionEmbedding, filtrosMetadata, safeLimit);

  return {
    items,
    mode: "legacy",
    warning: vectorSearchWarning,
  };
}

async function retrieveRegistrosContext(question, filtrosMetadata, limit) {
  const tableRaw = typeof env.ragRecordsTable === "string" ? env.ragRecordsTable.trim() : "";

  if (!tableRaw) {
    return {
      items: [],
      warning: "RAG_RECORDS_TABLE no esta configurada.",
    };
  }

  const safeTable = sanitizeIdentifier(tableRaw, {
    allowQualified: true,
    label: "RAG_RECORDS_TABLE",
  });
  const safeIdColumn = sanitizeIdentifier(env.ragRecordsIdColumn, {
    label: "RAG_RECORDS_ID_COLUMN",
  });
  const safeTextColumn = sanitizeIdentifier(env.ragRecordsTextColumn, {
    label: "RAG_RECORDS_TEXT_COLUMN",
  });

  let safeMetadataColumn = "";

  if (typeof env.ragRecordsMetadataColumn === "string" && env.ragRecordsMetadataColumn.trim()) {
    safeMetadataColumn = sanitizeIdentifier(env.ragRecordsMetadataColumn, {
      label: "RAG_RECORDS_METADATA_COLUMN",
    });
  }

  const sourceLabel = env.ragRecordsSourceLabel || safeTable;
  const queryParams = [question, `%${question}%`, sourceLabel];
  const conditions = [
    `(
      to_tsvector('spanish', COALESCE(${safeTextColumn}::text, '')) @@ websearch_to_tsquery('spanish', $1)
      OR COALESCE(${safeTextColumn}::text, '') ILIKE $2
    )`,
  ];

  let warning = "";

  if (Object.keys(filtrosMetadata).length > 0 && safeMetadataColumn) {
    queryParams.push(JSON.stringify(filtrosMetadata));
    conditions.push(`${safeMetadataColumn} @> $${queryParams.length}::jsonb`);
  } else if (Object.keys(filtrosMetadata).length > 0 && !safeMetadataColumn) {
    warning = "No se aplicaron filtrosMetadata en registros porque no hay columna metadata configurada.";
  }

  queryParams.push(limit);
  const metadataSelect = safeMetadataColumn ? `${safeMetadataColumn} AS metadata` : `'{}'::jsonb AS metadata`;

  const dbResult = await dbQuery(
    `
      SELECT
        ${safeIdColumn} AS id,
        COALESCE(${safeTextColumn}::text, '') AS contenido,
        $3::text AS fuente,
        ${metadataSelect},
        ts_rank_cd(
          to_tsvector('spanish', COALESCE(${safeTextColumn}::text, '')),
          websearch_to_tsquery('spanish', $1)
        ) AS score
      FROM ${safeTable}
      WHERE ${conditions.join(" AND ")}
      ORDER BY score DESC
      LIMIT $${queryParams.length}
    `,
    queryParams
  );

  return {
    items: dbResult.rows.map((row) => buildRecordContextItem(row, safeTable)),
    warning,
  };
}

function resolveDocField(doc, spanishKey, englishKey, fallback = "") {
  if (typeof doc[spanishKey] === "string") return doc[spanishKey].trim();
  if (typeof doc[englishKey] === "string") return doc[englishKey].trim();
  return fallback;
}

export async function ingestDocuments(documents = [], options = {}) {
  try {
    await ensureDocumentosRagSchema();

    const metadataGlobal = normalizeMetadata(options?.metadataGlobal);

    const normalizedDocs = documents
      .filter((doc) => doc && (typeof doc.contenido === "string" || typeof doc.content === "string"))
      .map((doc) => {
        const fuente = resolveDocField(doc, "fuente", "source", "manual");
        const contenido = resolveDocField(doc, "contenido", "content");
        const metadata = enrichDocumentMetadata(
          { ...metadataGlobal, ...normalizeMetadata(doc.metadata ?? doc.metadatos) },
          fuente,
          contenido
        );
        return { fuente, contenido, metadata };
      })
      .filter((doc) => doc.contenido.length > 0);

    if (!normalizedDocs.length) {
      return {
        ok: false,
        code: "BAD_REQUEST",
        error: "No hay documentos validos para procesar.",
      };
    }

    const inserted = [];

    for (const doc of normalizedDocs) {
      const embeddingRaw = await embedText(doc.contenido);
      const embedding = normalizeEmbeddingValues(embeddingRaw);
      const embeddingVector =
        embedding.length === env.embeddingDimensions ? toVectorLiteral(embedding) : null;

      const result = vectorSearchEnabled
        ? await dbQuery(
            `
              INSERT INTO documentos_rag (fuente, contenido, embedding, embedding_vector, metadata)
              VALUES ($1, $2, $3, $4::${vectorColumnType}, $5::jsonb)
              RETURNING id, fuente, metadata
            `,
            [
              doc.fuente,
              doc.contenido,
              JSON.stringify(embedding),
              embeddingVector,
              JSON.stringify(doc.metadata),
            ]
          )
        : await dbQuery(
            `
              INSERT INTO documentos_rag (fuente, contenido, embedding, metadata)
              VALUES ($1, $2, $3, $4::jsonb)
              RETURNING id, fuente, metadata
            `,
            [doc.fuente, doc.contenido, JSON.stringify(embedding), JSON.stringify(doc.metadata)]
          );

      inserted.push(result.rows[0]);
    }

    return {
      ok: true,
      ingested: inserted.length,
      documents: inserted.map((item) => ({
        id: item.id,
        fuente: item.fuente,
        metadata: parseMetadataRow(item.metadata),
      })),
    };
  } catch (error) {
    return buildServiceError(error, "No se pudo procesar la ingesta de documentos.");
  }
}

export async function queryRag(payload) {
  try {
    const startTime = Date.now();
    await ensureDocumentosRagSchema();

    const question = typeof payload?.question === "string" ? payload.question.trim() : "";
    const filtrosMetadata = normalizeMetadata(payload?.filtrosMetadata ?? payload?.metadataFiltro);
    const incluirRegistros = parseBoolean(
      payload?.incluirRegistros ?? payload?.incluirBaseDatos ?? payload?.usarRegistros
    );
    const requestedProvider = typeof payload?.provider === "string"
      ? payload.provider.trim().toLowerCase()
      : "";
    const topKGlobal = readPositiveInt(payload?.topK ?? payload?.limiteContexto, env.ragTopK);
    const topKDocumentosDefault = incluirRegistros ? env.ragTopKDocumentos : topKGlobal;
    const topKDocumentos = readPositiveInt(payload?.topKDocumentos, topKDocumentosDefault);
    const topKRegistros = incluirRegistros
      ? readPositiveInt(payload?.topKRegistros, env.ragTopKRegistros)
      : 0;

    if (!question) {
      return {
        ok: false,
        code: "BAD_REQUEST",
        error: "La pregunta no puede estar vacia.",
      };
    }

    const questionEmbedding = normalizeEmbeddingValues(await embedText(question));

    let documentsRetrieval = await retrieveDocumentContext(
      questionEmbedding,
      filtrosMetadata,
      topKDocumentos
    );

    let registrosRetrieval = {
      items: [],
      warning: "",
    };

    if (incluirRegistros && topKRegistros > 0) {
      try {
        registrosRetrieval = await retrieveRegistrosContext(question, filtrosMetadata, topKRegistros);
      } catch (recordError) {
        registrosRetrieval.warning =
          recordError instanceof Error
            ? recordError.message
            : "No se pudieron consultar registros de base de datos.";
      }
    }

    if (
      incluirRegistros &&
      registrosRetrieval.items.length === 0 &&
      documentsRetrieval.items.length < topKGlobal &&
      topKDocumentos < topKGlobal
    ) {
      documentsRetrieval = await retrieveDocumentContext(questionEmbedding, filtrosMetadata, topKGlobal);
    }

    const mergedContext = incluirRegistros
      ? [...documentsRetrieval.items, ...registrosRetrieval.items]
          .sort((a, b) => b.score - a.score)
          .slice(0, topKGlobal)
      : documentsRetrieval.items.slice(0, topKGlobal);

    const contextText = buildPromptContext(mergedContext);

    const systemInstruction = [
      "Responde únicamente con base en el contexto recuperado.",
      'Si no hay evidencia suficiente, responde: "No tengo evidencia suficiente en los documentos recuperados."',
      "Cita las fuentes utilizadas.",
      "Entrega solo la respuesta final para el usuario.",
      "No incluyas bloques internos como Question, Constraint, Context, Draft ni autoevaluaciones.",
      "Responde en español.",
    ].join("\n");

    const prompt = [
      systemInstruction,
      `Pregunta: ${question}`,
      `Contexto:\n${contextText || "Sin contexto disponible."}`,
    ].join("\n\n");

    let answer = "";
    let provider = "";
    let streamGenerator = null;
    const isStreaming = parseBoolean(payload?.stream);
    const includeContext = parseBoolean(
      payload?.includeContext ?? payload?.incluirContexto ?? payload?.debug ?? payload?.modoDebug
    );

    if (isStreaming) {
      if (requestedProvider === "groq") {
        streamGenerator = generateWithGroqStream(prompt);
        provider = "groq";
      } else {
        try {
          const llm = getLlmClient();
          const llmResponse = await llm.generateContentStream(prompt);
          streamGenerator = (async function* () {
            for await (const chunk of llmResponse.stream) {
              yield chunk.text();
            }
          })();
          provider = "google";
        } catch (geminiError) {
          streamGenerator = generateWithGroqStream(prompt);
          provider = "groq";
        }
      }
    } else {
      if (requestedProvider === "groq") {
        answer = await generateWithGroq(prompt);
        provider = "groq";
      } else {
        try {
          const llm = getLlmClient();
          const llmResponse = await llm.generateContent(prompt);
          answer = llmResponse.response.text();
          provider = "google";
        } catch (geminiError) {
          answer = await generateWithGroq(prompt);
          provider = "groq";

          if (!answer) {
            throw geminiError;
          }
        }
      }
    }

    if (!isStreaming) {
      answer = sanitizeModelAnswer(answer);
    }

    const latencyMs = Date.now() - startTime;

    const fragmentos = mergedContext.map(({ id, fuente, contenido, metadata, score, tipoFuente, identificador }) => ({
      id,
      fuente,
      contenido,
      metadata,
      score,
      tipoFuente,
      identificador,
    }));

    const responseObj = {
      ok: true,
      answer,
      fragmentosUsados: fragmentos,
      metadata: {
        embeddingModel: env.embeddingModel,
        llmModel: provider === "google" ? env.llmModel : env.groqModel,
        provider,
        esquema: "es",
        latencyMs,
        filtrosMetadata,
        retrieval: {
          documentos: {
            mode: documentsRetrieval.mode,
            requested: topKDocumentos,
            returned: documentsRetrieval.items.length,
            warning: documentsRetrieval.warning || undefined,
          },
          registros: {
            enabled: incluirRegistros,
            requested: incluirRegistros ? topKRegistros : 0,
            returned: registrosRetrieval.items.length,
            warning: registrosRetrieval.warning || undefined,
          },
          global: {
            requested: topKGlobal,
            returned: mergedContext.length,
          },
        },
      },
    };

    if (includeContext) {
      responseObj.context = fragmentos;
    }

    if (isStreaming && streamGenerator) {
      responseObj.stream = streamGenerator;
    }

    return responseObj;
  } catch (error) {
    return buildServiceError(error, "No se pudo procesar la consulta RAG.");
  }
}

export async function eliminarDocumentos(payload = {}) {
  try {
    await ensureDocumentosRagSchema();

    const fuente = typeof payload?.fuente === "string" ? payload.fuente.trim() : "";
    const metadata = normalizeMetadata(payload?.metadata ?? payload?.metadatos);

    if (!fuente && Object.keys(metadata).length === 0) {
      return {
        ok: false,
        code: "BAD_REQUEST",
        error: "Debes enviar al menos 'fuente' o 'metadata' para eliminar documentos.",
      };
    }

    const conditions = [];
    const params = [];

    if (fuente) {
      params.push(fuente);
      conditions.push(`fuente = $${params.length}`);
    }

    if (Object.keys(metadata).length > 0) {
      params.push(JSON.stringify(metadata));
      conditions.push(`metadata @> $${params.length}::jsonb`);
    }

    const deleted = await dbQuery(
      `
        DELETE FROM documentos_rag
        WHERE ${conditions.join(" AND ")}
        RETURNING id
      `,
      params
    );

    return {
      ok: true,
      eliminados: deleted.rowCount || 0,
    };
  } catch (error) {
    return buildServiceError(error, "No se pudieron eliminar documentos.");
  }
}

export async function getStats() {
  try {
    await ensureDocumentosRagSchema();

    const countResult = await dbQuery(
      "SELECT COUNT(*) AS total FROM documentos_rag"
    );

    const sourcesResult = await dbQuery(
      "SELECT COUNT(DISTINCT fuente) AS total FROM documentos_rag"
    );

    return {
      ok: true,
      totalDocumentos: Number(countResult.rows[0]?.total) || 0,
      totalFuentes: Number(sourcesResult.rows[0]?.total) || 0,
      vectorSearchEnabled,
      config: {
        embeddingModel: env.embeddingModel,
        embeddingDimensions: env.embeddingDimensions,
        llmModel: env.llmModel,
        groqModel: env.groqModel,
        chunkSize: env.ragChunkSize,
        chunkOverlap: env.ragChunkOverlap,
        topK: env.ragTopK,
        vectorColumnType,
        vectorIndexMethod,
      },
    };
  } catch (error) {
    return buildServiceError(error, "No se pudieron obtener estadisticas.");
  }
}
