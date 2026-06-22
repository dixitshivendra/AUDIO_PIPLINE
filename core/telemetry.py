import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor


def setup_telemetry(service_name: str):
    # Setup Resource identification
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Configure OTLP Exporter (Sending traces to Jaeger)
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://jaeger:4317")
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    # Add batch processor for performance
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Inject Trace IDs into your existing JSON logs
    LoggingInstrumentor().instrument(set_logging_format=True)
