const fs = require('fs');
const zlib = require('zlib');

function crc32(buf) {
  let c;
  let table = [];
  for (let n = 0; n < 256; n++) {
    c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c;
  }
  let crc = 0 ^ (-1);
  for (let i = 0; i < buf.length; i++) {
    crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
  }
  return (crc ^ (-1)) >>> 0;
}

function makeChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.alloc(4);
  const toHash = Buffer.concat([typeBuf, data]);
  crcBuf.writeUInt32BE(crc32(toHash), 0);
  return Buffer.concat([len, typeBuf, data, crcBuf]);
}

const fileBuf = fs.readFileSync('frontend/public/robot_original.png');
const width = fileBuf.readUInt32BE(16);
const height = fileBuf.readUInt32BE(20);

// Extract IDAT chunks
let idatChunks = [];
let offset = 8;
while (offset < fileBuf.length) {
  const len = fileBuf.readUInt32BE(offset);
  const type = fileBuf.toString('ascii', offset + 4, offset + 8);
  if (type === 'IDAT') {
    idatChunks.push(fileBuf.subarray(offset + 8, offset + 8 + len));
  }
  offset += 12 + len;
}

const compressed = Buffer.concat(idatChunks);
const decompressed = zlib.inflateSync(compressed);

const stride = 1 + width * 4;
// Unfilter scanlines (standard PNG unfilter for RGBA)
const raw = Buffer.alloc(height * width * 4);

let pos = 0;
for (let y = 0; y < height; y++) {
  const filter = decompressed[pos++];
  for (let x = 0; x < width; x++) {
    const rawIdx = (y * width + x) * 4;
    let r = decompressed[pos++];
    let g = decompressed[pos++];
    let b = decompressed[pos++];
    let a = decompressed[pos++];

    const leftIdx = (x > 0) ? ((y * width + x - 1) * 4) : -1;
    const upIdx = (y > 0) ? (((y - 1) * width + x) * 4) : -1;
    const upLeftIdx = (y > 0 && x > 0) ? (((y - 1) * width + x - 1) * 4) : -1;

    if (filter === 1) { // Sub
      if (leftIdx >= 0) {
        r = (r + raw[leftIdx]) & 0xff;
        g = (g + raw[leftIdx + 1]) & 0xff;
        b = (b + raw[leftIdx + 2]) & 0xff;
        a = (a + raw[leftIdx + 3]) & 0xff;
      }
    } else if (filter === 2) { // Up
      if (upIdx >= 0) {
        r = (r + raw[upIdx]) & 0xff;
        g = (g + raw[upIdx + 1]) & 0xff;
        b = (b + raw[upIdx + 2]) & 0xff;
        a = (a + raw[upIdx + 3]) & 0xff;
      }
    } else if (filter === 3) { // Average
      const lR = leftIdx >= 0 ? raw[leftIdx] : 0;
      const uR = upIdx >= 0 ? raw[upIdx] : 0;
      const lG = leftIdx >= 0 ? raw[leftIdx + 1] : 0;
      const uG = upIdx >= 0 ? raw[upIdx + 1] : 0;
      const lB = leftIdx >= 0 ? raw[leftIdx + 2] : 0;
      const uB = upIdx >= 0 ? raw[upIdx + 2] : 0;
      const lA = leftIdx >= 0 ? raw[leftIdx + 3] : 0;
      const uA = upIdx >= 0 ? raw[upIdx + 3] : 0;
      r = (r + Math.floor((lR + uR) / 2)) & 0xff;
      g = (g + Math.floor((lG + uG) / 2)) & 0xff;
      b = (b + Math.floor((lB + uB) / 2)) & 0xff;
      a = (a + Math.floor((lA + uA) / 2)) & 0xff;
    } else if (filter === 4) { // Paeth
      const paeth = (a, b, c) => {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        if (pb <= pc) return b;
        return c;
      };
      const lR = leftIdx >= 0 ? raw[leftIdx] : 0;
      const uR = upIdx >= 0 ? raw[upIdx] : 0;
      const ulR = upLeftIdx >= 0 ? raw[upLeftIdx] : 0;
      const lG = leftIdx >= 0 ? raw[leftIdx + 1] : 0;
      const uG = upIdx >= 0 ? raw[upIdx + 1] : 0;
      const ulG = upLeftIdx >= 0 ? raw[upLeftIdx + 1] : 0;
      const lB = leftIdx >= 0 ? raw[leftIdx + 2] : 0;
      const uB = upIdx >= 0 ? raw[upIdx + 2] : 0;
      const ulB = upLeftIdx >= 0 ? raw[upLeftIdx + 2] : 0;
      const lA = leftIdx >= 0 ? raw[leftIdx + 3] : 0;
      const uA = upIdx >= 0 ? raw[upIdx + 3] : 0;
      const ulA = upLeftIdx >= 0 ? raw[upLeftIdx + 3] : 0;
      r = (r + paeth(lR, uR, ulR)) & 0xff;
      g = (g + paeth(lG, uG, ulG)) & 0xff;
      b = (b + paeth(lB, uB, ulB)) & 0xff;
      a = (a + paeth(lA, uA, ulA)) & 0xff;
    }

    raw[rawIdx] = r;
    raw[rawIdx + 1] = g;
    raw[rawIdx + 2] = b;
    raw[rawIdx + 3] = a;
  }
}

// Chroma Key Green Background Removal
for (let i = 0; i < raw.length; i += 4) {
  const r = raw[i];
  const g = raw[i + 1];
  const b = raw[i + 2];

  // Detect chroma green (#00d000 - #00ff00)
  if (g > 85 && g > r * 1.25 && g > b * 1.25) {
    const diff = g - Math.max(r, b) * 1.25;
    if (diff > 30) {
      raw[i + 3] = 0; // Fully transparent
    } else {
      raw[i + 3] = Math.max(0, Math.min(255, Math.floor(255 - (diff / 30) * 255)));
    }
  } else if (g > r && g > b && g > 60) {
    // Suppress green fringe reflections on white porcelain
    raw[i + 1] = Math.round((r + b) / 2);
  }
}

// Re-encode to PNG using filter 0
const newDecompressed = Buffer.alloc(height * (1 + width * 4));
let wPos = 0;
let rPos = 0;
for (let y = 0; y < height; y++) {
  newDecompressed[wPos++] = 0; // Filter None
  for (let x = 0; x < width * 4; x++) {
    newDecompressed[wPos++] = raw[rPos++];
  }
}

const newCompressed = zlib.deflateSync(newDecompressed, { level: 9 });

// Assemble new PNG
const header = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(width, 0);
ihdr.writeUInt32BE(height, 4);
ihdr[8] = 8; // bit depth
ihdr[9] = 6; // color type RGBA
ihdr[10] = 0;
ihdr[11] = 0;
ihdr[12] = 0;

const ihdrChunk = makeChunk('IHDR', ihdr);
const idatChunk = makeChunk('IDAT', newCompressed);
const iendChunk = makeChunk('IEND', Buffer.alloc(0));

const outPng = Buffer.concat([header, ihdrChunk, idatChunk, iendChunk]);
fs.writeFileSync('frontend/public/robot_transparent.png', outPng);
console.log('Successfully created frontend/public/robot_transparent.png! Size:', outPng.length, 'bytes');

