
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool, FunctionDeclaration, Content, FinishReason
from typing import AsyncIterator, Optional, Any
from ..types import (
    ModelDef, Context, AssistantMessage,
    StreamEvent, TextContent, ToolCallContent as ToolCall,
    StreamOptions, Message
)
from ..utils.partial_json import parse_streaming_json
from ..transform_messages import transform_messages

class GoogleVertexProvider:
    def __init__(self):
        self.api = "google-vertex"

    async def stream(
        self,
        model: ModelDef,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AsyncIterator[StreamEvent]:
        options = options or StreamOptions()
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        vertexai.init(project=project_id, location=location)

        # Convert tools
        vertex_tools = []
        if context.tools:
            declarations = []
            for t in context.tools:
                declarations.append(
                    FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=t.parameters
                    )
                )
            if declarations:
                vertex_tools.append(Tool(function_declarations=declarations))
        
        # System instructions
        system_instruction = context.system_prompt if context.system_prompt else None

        # Initialize model
        gen_model = GenerativeModel(
            model_name=model.id,
            system_instruction=system_instruction,
            tools=vertex_tools
        )

        # Convert messages
        transformed = transform_messages(context.messages, model)
        history = self._convert_messages(transformed)

        # Generation config
        generation_config = {
            "max_output_tokens": options.max_tokens or model.max_output,
            "temperature": options.temperature or 0.5,
        }

        # Send request
        # Note: vertexai python SDK handles history slightly differently (chat session)
        # But convert_messages returns Content list which can be passed to generate_content
        
        response_stream = await gen_model.generate_content_async(
            history,
            generation_config=generation_config,
            stream=True
        )

        output_message = AssistantMessage(
            role="assistant",
            content=[],
            provider=model.provider,
            model=model.id,
        )

        current_block_index = 0

        async for chunk in response_stream:
            # Yield start if first chunk
            if output_message.content == [] and not output_message.tool_calls: # heuristic
                 pass # we yield start explicitly below if needed, or imply it

            for part in chunk.candidates[0].content.parts:
                if part.text:
                    # Check if we are already building a text block
                    if not output_message.content or not isinstance(output_message.content[-1], TextContent):
                        output_message.content.append(TextContent(text=""))
                        current_block_index = len(output_message.content) - 1
                        yield StreamEvent(type="text_start", content_index=current_block_index, partial=output_message)

                    output_message.content[-1].text += part.text
                    yield StreamEvent(type="text_delta", content_index=current_block_index, delta=part.text, partial=output_message)
                
                if part.function_call:
                    fc = part.function_call
                    tool_call = ToolCall(
                        id=f"{fc.name}_{int(time.time())}", # Vertex doesn't give stable IDs easily in all versions
                        name=fc.name,
                        arguments=dict(fc.args) # Args come fully parsed in Vertex stream usually
                    )
                    
                    # Vertex sends full function call in one go usually, but let's handle it
                    output_message.content.append(tool_call)
                    current_block_index = len(output_message.content) - 1
                    yield StreamEvent(type="toolcall_start", content_index=current_block_index, partial=output_message)
                    yield StreamEvent(type="toolcall_end", content_index=current_block_index, tool_call=tool_call, partial=output_message)

            # Metadata/Usage
            if chunk.usage_metadata:
                output_message.usage.input_tokens = chunk.usage_metadata.prompt_token_count
                output_message.usage.output_tokens = chunk.usage_metadata.candidates_token_count
                output_message.usage.total_tokens = chunk.usage_metadata.total_token_count

        yield StreamEvent(type="done", reason="stop", message=output_message)
        
    def _convert_messages(self, messages: list[Message]) -> list[Content]:
        # Implementation of message conversion to Vertex Content types
        # This needs to be robust mapping user/assistant/tool roles
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            parts = []
            if isinstance(m.content, str):
                parts.append(Part.from_text(m.content))
            # ... handle other types
            contents.append(Content(role=role, parts=parts))
        return contents
