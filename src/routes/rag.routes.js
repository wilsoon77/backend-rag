import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { eliminarDocumentos, getStats, ingestDocuments, queryRag } from "../services/rag.service.js";
import { extractChunksFromFiles } from "../services/file-ingest.service.js";
import {
  isFileLike,
  statusFromCode,
  parseMetadataInput,
  parseBoolean,
} from "../utils/helpers.js";

const ragRouter = new Hono();

ragRouter.post("/ingest", async (c) => {
  const body = await c.req.json().catch(() => null);

  if (!body || !Array.isArray(body.documents)) {
    return c.json(
      {
        error: "Invalid request",
        message: "Debes enviar un JSON con el campo 'documents' (array).",
      },
      400
    );
  }

  const metadataParsed = parseMetadataInput(body.metadata ?? body.metadatos);

  if (!metadataParsed.ok) {
    return c.json(
      {
        error: "Invalid request",
        message: metadataParsed.message,
      },
      400
    );
  }

  const result = await ingestDocuments(body.documents, {
    metadataGlobal: metadataParsed.value,
  });
  return c.json(result, result.ok ? 200 : statusFromCode(result.code));
});

ragRouter.post("/ingest/archivo", async (c) => {
  const formData = await c.req.formData().catch(() => null);

  if (!formData) {
    return c.json(
      {
        error: "Invalid request",
        message: "Debes enviar multipart/form-data con archivos en 'files' o 'file'.",
      },
      400
    );
  }

  const filesFromList = formData.getAll("files");
  const singleFile = formData.get("file");
  const candidates = filesFromList.length > 0 ? filesFromList : singleFile ? [singleFile] : [];
  const files = candidates.filter(isFileLike);

  if (files.length === 0) {
    return c.json(
      {
        error: "Invalid request",
        message: "No se encontraron archivos validos. Usa 'files' o 'file'.",
      },
      400
    );
  }

  const fuenteBaseRaw = formData.get("fuente");
  const metadataParsed = parseMetadataInput(formData.get("metadata") ?? formData.get("metadatos"));

  if (!metadataParsed.ok) {
    return c.json(
      {
        error: "Invalid request",
        message: metadataParsed.message,
      },
      400
    );
  }

  const metadataBase = {
    ...metadataParsed.value,
  };

  const cursoRaw = formData.get("curso");
  const materiaRaw = formData.get("materia");

  if (typeof cursoRaw === "string" && cursoRaw.trim()) {
    metadataBase.curso = cursoRaw.trim();
  }

  if (typeof materiaRaw === "string" && materiaRaw.trim()) {
    metadataBase.materia = materiaRaw.trim();
  }

  const reemplazar = parseBoolean(formData.get("reemplazar"));
  let eliminadosPrevios = 0;

  if (reemplazar) {
    for (const file of files) {
      const deleteCriteria = {
        archivo: file.name,
      };

      if (metadataBase.curso) {
        deleteCriteria.curso = metadataBase.curso;
      }

      if (metadataBase.materia) {
        deleteCriteria.materia = metadataBase.materia;
      }

      const deletionResult = await eliminarDocumentos({ metadata: deleteCriteria });

      if (!deletionResult.ok) {
        return c.json(deletionResult, statusFromCode(deletionResult.code));
      }

      eliminadosPrevios += deletionResult.eliminados;
    }
  }

  const extraction = await extractChunksFromFiles(files, {
    fuenteBase: typeof fuenteBaseRaw === "string" ? fuenteBaseRaw : "",
    metadataBase,
  });

  if (!extraction.ok) {
    return c.json(extraction, statusFromCode(extraction.code));
  }

  const result = await ingestDocuments(extraction.documents);

  if (!result.ok) {
    return c.json(result, statusFromCode(result.code));
  }

  return c.json(
    {
      ok: true,
      ingested: result.ingested,
      documentos: result.documents,
      archivos: extraction.archivos,
      totalFragmentos: extraction.totalFragmentos,
      eliminadosPrevios,
      config: extraction.config,
    },
    200
  );
});

ragRouter.post("/documentos/eliminar", async (c) => {
  const body = await c.req.json().catch(() => null);

  if (!body || typeof body !== "object") {
    return c.json(
      {
        error: "Invalid request",
        message: "Debes enviar un JSON valido.",
      },
      400
    );
  }

  const metadataParsed = parseMetadataInput(body.metadata ?? body.metadatos);

  if (!metadataParsed.ok) {
    return c.json(
      {
        error: "Invalid request",
        message: metadataParsed.message,
      },
      400
    );
  }

  const result = await eliminarDocumentos({
    fuente: body.fuente,
    metadata: metadataParsed.value,
  });

  return c.json(result, result.ok ? 200 : statusFromCode(result.code));
});

ragRouter.post("/query", async (c) => {
  const body = await c.req.json().catch(() => null);

  if (!body || typeof body.question !== "string") {
    return c.json(
      {
        error: "Invalid request",
        message: "Debes enviar un JSON con el campo 'question' (string).",
      },
      400
    );
  }

  const result = await queryRag(body);

  if (!result.ok) {
    return c.json(result, statusFromCode(result.code));
  }

  if (body.stream && result.stream) {
    return streamSSE(c, async (stream) => {
      // Send sources and metadata early so the UI can construct citations.
      const metadataEvent = {
        fragmentosUsados: result.fragmentosUsados,
        metadata: result.metadata,
      };

      if (Array.isArray(result.context)) {
        metadataEvent.context = result.context;
      }

      await stream.writeSSE({
        event: "metadata",
        data: JSON.stringify(metadataEvent),
      });

      // Send the text chunks
      for await (const chunk of result.stream) {
        await stream.writeSSE({
          data: JSON.stringify({ chunk }),
        });
      }

      await stream.writeSSE({
        event: "end",
        data: "[DONE]",
      });
    });
  }

  return c.json(result, 200);
});

ragRouter.get("/stats", async (c) => {
  const result = await getStats();
  return c.json(result, result.ok ? 200 : statusFromCode(result.code));
});

export default ragRouter;
