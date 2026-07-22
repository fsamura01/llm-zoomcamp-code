from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from rag_logger import SQLiteSpanExporter

from evaluation_utils import calc_price
from rag_helper import RAGBase

# Set up OTel with console output
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("logs/traces.db")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")


# Create a traced version of RAG
class RAGTraced(RAGBase):

    def search(self, query):
        with tracer.start_as_current_span("search") as span:
            results = super().search(query)
            return results

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("cost", calc_price(usage)["total_cost"])

            # Cost calculation (openai/gpt-oss-20b pricing)
            #cost = calc_price(usage)
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            answer = super().rag(query)
            return answer
