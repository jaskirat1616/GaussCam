/* WebGPU renderer that consumes backend Gaussian payloads. */

type FloatArray = Float32Array | number[];

export interface WebGPUPayload {
  centroids: number[][];
  scales: number[][] | null;
  rotations: number[][] | null;
  colors: number[][];
  opacity: number[];
  camera_pose?: number[][];
  intrinsics?: number[][];
  resolution: [number, number];
  motion?: {
    translation: number[][];
    rotation_axis_angle: number[][];
    scale_velocity: number[][];
    timestamps: number[];
  };
}

export class WebGPUGaussianRenderer {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private payload: WebGPUPayload | null = null;

  async initialize(canvas: HTMLCanvasElement): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser");
    }

    this.adapter = await navigator.gpu.requestAdapter();
    if (!this.adapter) {
      throw new Error("Failed to get WebGPU adapter");
    }

    this.device = await this.adapter.requestDevice();
    this.context = canvas.getContext("webgpu");
    if (!this.context) {
      throw new Error("Failed to get WebGPU canvas context");
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format });
    this.pipeline = this.createPipeline(format);
  }

  loadPayload(payload: WebGPUPayload): void {
    this.payload = payload;
  }

  render(): void {
    if (!this.device || !this.context || !this.pipeline || !this.payload) {
      return;
    }

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    // TODO: Upload Gaussian payload to GPU buffers for rendering.
    renderPass.draw(3, 1, 0, 0);
    renderPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  private createPipeline(format: GPUTextureFormat): GPURenderPipeline {
    if (!this.device) {
      throw new Error("Device not initialized");
    }

    const shaderModule = this.device.createShaderModule({
      code: `
        @vertex
        fn vs_main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
          var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(3.0, -1.0),
            vec2<f32>(-1.0, 3.0)
          );
          let uv = pos[VertexIndex];
          return vec4<f32>(uv, 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
          return vec4<f32>(0.1, 0.1, 0.1, 1.0);
        }
      `,
    });

    return this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
  }
}

export function supportsWebGPU(): boolean {
  return typeof navigator !== "undefined" && Boolean(navigator.gpu);
}

