import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';

const SUPABASE_BUCKET = 'insuranceRISKagent';

// ── Supabase S3 ───────────────────────────────────────────────────────────────

function _supabaseClient() {
  const endpoint        = process.env.SUPABASE_S3_ENDPOINT;
  const region          = process.env.SUPABASE_S3_REGION;
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
 * Upload a text file to Supabase Storage (S3-compatible endpoint).
 * @param {string} key         - Object key e.g. "equity-research/NVDA-2026-06-25.md"
 * @param {string} body        - File content
 * @param {string} contentType - Defaults to text/markdown
 */
export async function uploadFile(key, body, contentType = 'text/markdown; charset=utf-8') {
  const client = _supabaseClient();
  if (!client) {
    console.error('[supabase-s3] credentials not set — skipping upload');
    return;
  }
  await client.send(new PutObjectCommand({
    Bucket:      SUPABASE_BUCKET,
    Key:         key,
    Body:        body,
    ContentType: contentType,
  }));
  console.error(`[supabase-s3] uploaded → ${SUPABASE_BUCKET}/${key}`);
}

// ── AWS S3 ────────────────────────────────────────────────────────────────────

function _awsClient() {
  const accessKeyId     = process.env.AWS_ACCESS_KEY_ID;
  const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;
  const region          = process.env.AWS_S3_REGION ?? 'eu-west-1';

  if (!accessKeyId || !secretAccessKey) return null;

  return new S3Client({ region, credentials: { accessKeyId, secretAccessKey } });
}

/**
 * Upload a text file to AWS S3.
 * @param {string} key         - Object key e.g. "equity-research/NVDA-2026-06-25.md"
 * @param {string} body        - File content
 * @param {string} contentType - Defaults to text/markdown
 */
export async function uploadFileToAWS(key, body, contentType = 'text/markdown; charset=utf-8') {
  const client = _awsClient();
  if (!client) {
    console.error('[aws-s3] AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — skipping upload');
    return;
  }
  const bucket = process.env.AWS_S3_BUCKET;
  if (!bucket) {
    console.error('[aws-s3] AWS_S3_BUCKET not set — skipping upload');
    return;
  }
  await client.send(new PutObjectCommand({
    Bucket:      bucket,
    Key:         key,
    Body:        body,
    ContentType: contentType,
  }));
  console.error(`[aws-s3] uploaded → ${bucket}/${key}`);
}
