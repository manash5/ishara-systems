import { sql } from "drizzle-orm";
import { drizzle, type PostgresJsDatabase } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { env } from "../config/constant";
import * as schema from "../models";

export type Database = PostgresJsDatabase<typeof schema>;

let client: postgres.Sql | null = null;
let db: Database | null = null;

/** Lazily creates the shared connection pool. `prepare: false` is required by the Supabase pooler. */
export function getDb(): Database {
  if (!db) {
    client = postgres(env.DATABASE_URL, { prepare: false });
    db = drizzle(client, { schema, casing: "snake_case" });
  }
  return db;
}

/** Throws if the database cannot be reached. */
export async function pingDb(): Promise<void> {
  await getDb().execute(sql`select 1`);
}

export async function closeDb(): Promise<void> {
  if (client) {
    await client.end({ timeout: 5 });
    client = null;
    db = null;
  }
}
