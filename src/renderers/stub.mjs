/* A renderer that does not render.

   It exists so the whole loop — pair, benchmark, connect, receive a job, upload
   a result, get paid — can be exercised without the model, and so the native
   renderer has an interface to slot into. Every renderer is:

     { name, accelerator, async warm(), async render(job, onProgress) -> Buffer }
*/
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const JPEG = Buffer.from(readFileSync(join(here, "stub.jpg.b64"), "utf8"), "base64");

export function createStubRenderer() {
  return {
    name: "stub",
    accelerator: "none (stub)",
    async warm() {},
    async render(job, onProgress = () => {}) {
      /* Pretend to work, roughly in proportion to the steps asked for, so the
         dispatcher's timeouts and the queue behaviour get exercised honestly. */
      const perStep = 60;
      for (let step = 1; step <= job.steps; step++) {
        await new Promise((r) => setTimeout(r, perStep));
        onProgress(step, job.steps);
      }
      return JPEG;
    },
  };
}
