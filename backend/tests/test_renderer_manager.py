from backend.renderer.manager import create_renderer


def test_create_renderer_webgpu_fallback():
    renderer = create_renderer(preferred="webgpu")
    assert renderer.capabilities["web_stream"] is True

