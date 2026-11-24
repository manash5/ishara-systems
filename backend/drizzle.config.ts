import { config } from 'dotenv';
import { defineConfig } from 'drizzle-kit';

config({ path: ['.env.local', '.env'], quiet: true });

if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL is not set (expected in backend/.env.local or backend/.env)');
}

export default defineConfig({
  dialect: 'postgresql',
  schema: './src/models/index.ts',
  out: './drizzle',
  casing: 'snake_case',
  dbCredentials: { url: process.env.DATABASE_URL },
});
