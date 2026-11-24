import type { Request, Response } from "express";
import { ApiResponseHelper } from "../utils/api-response";

export function notFoundMiddleware(req: Request, res: Response) {
  return ApiResponseHelper.error(res, 404, `Route ${req.method} ${req.path} not found`);
}
