import type { NextFunction, Request, Response } from "express";
import { ZodError } from "zod";
import { HttpException } from "../exceptions/http-exception";
import { ApiResponseHelper } from "../utils/api-response";

/** Shape of errors thrown by express.json() (body-parser / http-errors). */
interface BodyParserError extends Error {
  type?: string;
  status?: number;
}

function isBodyParserError(err: unknown): err is BodyParserError {
  return err instanceof Error && typeof (err as BodyParserError).type === "string";
}

// Express 5 forwards thrown/rejected errors from async handlers here. Must keep 4 params.
export function errorMiddleware(err: unknown, _req: Request, res: Response, _next: NextFunction) {
  if (err instanceof HttpException) {
    return ApiResponseHelper.error(res, err.status, err.message);
  }

  if (err instanceof ZodError) {
    const issue = err.issues[0];
    const path = issue?.path.join(".");
    const message = issue ? (path ? `${path}: ${issue.message}` : issue.message) : "Invalid input";
    return ApiResponseHelper.error(res, 400, message);
  }

  if (isBodyParserError(err)) {
    if (err.type === "entity.parse.failed") return ApiResponseHelper.error(res, 400, "Malformed JSON body");
    if (err.type === "entity.too.large") return ApiResponseHelper.error(res, 413, "Request body too large");
    if (err.status && err.status >= 400 && err.status < 500) {
      return ApiResponseHelper.error(res, err.status, err.message);
    }
  }

  console.error(err);
  return ApiResponseHelper.error(res, 500, "Internal server error");
}
