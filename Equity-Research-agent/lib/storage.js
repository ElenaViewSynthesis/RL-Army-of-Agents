import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';

const BUCKET = 'insuranceRISKagent';

function _client() {
  const endpoint  = process.env.SUPABASE_S3_ENDPOINT;
  const region    = process.env.SUPABASE_S3_REGION;
  const accessKeyId     = process.env.SUPABASE_S3_ACCESS_KEY;
  const secretAccessKey = process.env.SUPABASE_S3_SECRET_KEY;

  if (!endpoint || !accessKeyId || !secretAccessKey) return null;

  return new S3Client({
    forcePathStyle: true,
    region: region ?? 'eu-west-1',
    endpoint,
    credentials: { accessKeyId, secretAccessKey },
  });
}

/**
 * Upload a text file to Supabase Storage (S3 protocol).
 * @param {string} key    - Object key, e.g. "equity-research/NVDA-2026-06-25.md"
 * @param {string} body   - File content
 * @param {string} [contentType] - Defaults to text/markdown
 */
export async function uploadFile(key, body, contentType = 'text/markdown; charset=utf-8') {
  const client = _client();
  if (!client) {
    console.error('[storage] S3 credentials not set — skipping upload');
    return;
  }
  await client.send(new PutObjectCommand({
    Bucket:      BUCKET,
    Key:         key,
    Body:        body,
    ContentType: contentType,
  }));
  console.error(`[storage] uploaded → ${BUCKET}/${key}`);
}
