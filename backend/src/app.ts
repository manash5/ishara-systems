import cors from "cors";
import express, { type Request, type Response } from "express";
import { rateLimit } from "express-rate-limit";
import helmet from "helmet";
import { env } from "./config/constant";
import { pingDb } from "./database/db";
import { HttpException } from "./exceptions/http-exception";
import { errorMiddleware } from "./middleware/error.middleware";
import { notFoundMiddleware } from "./middleware/not-found.middleware";
import { ApiResponseHelper } from "./utils/api-response";

const app = express();

app.disable("x-powered-by");
app.set("trust proxy", 1);
app.use(helmet());
app.use(cors({ origin: env.CORS_ORIGIN ?? true }));
app.use(express.json({ limit: "100kb" }));
app.use(
  "/api",
  rateLimit({
    windowMs: 60_000,
    limit: 120,
    standardHeaders: "draft-8",
    legacyHeaders: false,
    handler: (_req, res) => ApiResponseHelper.error(res, 429, "Too many requests, please try again later"),
  }),
);

app.get("/api/v1/health", async (_req: Request, res: Response) => {
  try {
    await pingDb();
  } catch (err) {
    console.error("Health check: database unreachable", err);
    throw new HttpException(503, "Database unavailable");
  }
  return ApiResponseHelper.success(res, { uptime: Math.round(process.uptime()), database: "up" }, 200, "OK");
});

// Feature routers are mounted here: app.use("/api/v1/<resource>", route);

app.use(notFoundMiddleware);
app.use(errorMiddleware);

export default app;
