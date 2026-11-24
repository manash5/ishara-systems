import type { Response } from "express";

export interface PaginationMeta {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

export class ApiResponseHelper {
  static success<T>(res: Response, data: T, status = 200, message = "Success", meta?: PaginationMeta) {
    return res.status(status).json({ status, success: true, message, data, ...(meta ? { meta } : {}) });
  }

  static error(res: Response, status: number, message: string) {
    return res.status(status).json({ status, success: false, message });
  }
}
