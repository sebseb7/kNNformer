import { pipeline, env } from '@xenova/transformers';
import HNSWLib from 'hnswlib-node';

env.cacheDir = './transformers-cache';

const DIM = 384; // MiniLM embedding size
const NUM_PRODUCTS = 7000;

// Cosine similarity index
const space = 'cosine';

function generateMockProducts(n) {
  const brands = ["Apple", "Samsung", "Google", "Sony", "Dell", "Nike", "Adidas"];
  const types = ["Phone", "Laptop", "Headphones", "Shoes", "Camera", "Monitor"];
  const products = [];

  for (let i = 0; i < n; i++) {
    const brand = brands[Math.floor(Math.random() * brands.length)];
    const type = types[Math.floor(Math.random() * types.length)];
    const model = Math.floor(Math.random() * 1000);
    products.push({
      id: i,
      brand,
      type,
      model,
      name: `${brand} ${type} Model ${model}`
    });
  }
  return products;
}

(async () => {
  const products = generateMockProducts(NUM_PRODUCTS);
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  console.log("Batch embedding products...");
  const batchSize = 128;
  const batches = [];

  for (let i = 0; i < products.length; i += batchSize) {
    batches.push(products.slice(i, i + batchSize).map(p => p.name));
  }

  const allEmbeddings = [];
  for (const batch of batches) {
    const result = await embedder(batch, { pooling: 'mean', normalize: true });
    for (const r of result) {
      allEmbeddings.push(Float32Array.from(r.data));
    }
  }

  console.log("Building HNSW index...");
  const index = new HNSWLib.HierarchicalNSW(space, DIM);
  index.initIndex(NUM_PRODUCTS);
  for (let i = 0; i < allEmbeddings.length; i++) {
    index.addPoint(Array.from(allEmbeddings[i]), i);
  }

  console.log("Index built.");

  const userQuery = "cam aple";
  const queryVec = await embedder(userQuery, { pooling: 'mean', normalize: true });
  const queryEmbedding = Float32Array.from(queryVec.data);

  const k = 5; // or however many top results you want
  const result = index.searchKnn(Array.from(queryEmbedding), k);

  const resultsWithMetadata = result.neighbors.map((neighborIndex, i) => {
    const distance = result.distances[i];
    const similarity = 1 - distance;
    return {
      similarity: similarity.toFixed(4),
      ...products[neighborIndex], // full product object
    };
  });

  resultsWithMetadata.sort((a, b) => b.similarity - a.similarity);

  console.log(`Top ${k} matches:\n`);
  for (const product of resultsWithMetadata) {
    console.log(product);
  }

})();