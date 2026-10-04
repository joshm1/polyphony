import { z } from 'zod'

import { PolyphonyDataSchema } from './types'
import type { PolyphonyData } from './types'

async function parseResponse<S extends z.ZodType>(r: Response, schema: S): Promise<z.output<S>> {
  const body: unknown = await r.json()
  const parsed = schema.safeParse(body)
  if (!parsed.success) {
    throw new Error(`Unexpected response from ${r.url}: ${z.prettifyError(parsed.error)}`)
  }
  return parsed.data
}

export async function postJson<S extends z.ZodType>(
  url: string,
  body: unknown,
  schema: S
): Promise<z.output<S>> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
  return parseResponse(r, schema)
}

export async function fetchData(): Promise<PolyphonyData> {
  const r = await fetch('/api/data')
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return parseResponse(r, PolyphonyDataSchema)
}
