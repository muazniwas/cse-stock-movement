/* eslint-disable @typescript-eslint/no-explicit-any */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE;

if (!API_BASE) {
  console.warn("NEXT_PUBLIC_API_BASE is not set. Add it to frontend/.env.local");
}

export class ApiError extends Error {
  status: number;
  detail?: any;

  constructor(message: string, status: number, detail?: any) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function parseMaybeJson(res: Response) {
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;

  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const body = await parseMaybeJson(res);

  if (!res.ok) {
    const message =
      typeof body === "object" && body && "detail" in body
        ? String((body as any).detail)
        : `Request failed: ${res.status}`;
    throw new ApiError(message, res.status, body);
  }

  return body as T;
}
