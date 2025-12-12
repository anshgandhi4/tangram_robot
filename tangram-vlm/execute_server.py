import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from google import genai
from google.genai import types

# Configuration
MCP_SERVER_URL = "http://localhost:8000/sse"
API_KEY = os.environ.get("GEMINI_API_KEY")

async def run_planning_cycle(user_instruction):
    client = genai.Client(api_key=API_KEY)

    print(f"connecting at {MCP_SERVER_URL}...")
    async with sse_client(MCP_SERVER_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

            tools = await session.list_tools()

            print(tools)
            funcs = []
            for tool in tools.tools:
                funcs.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                })

            chat = client.chats.create(model="gemini-3-pro-preview",
                                       config=types.GenerateContentConfig(
                                        temperature=1,
                                        tools=[{"function_declarations": funcs}]
                                    )
                                )

            system_prompt = """
you are looking at a set of tangram pieces. try to arrange them to form your best approxmation of whatever task the user is interested in, using all MCP tools available to you! You may do/undo moves as you wish, as long as the final configuration is an approximation to whatever the user wants, with no overlaps between pieces. After you're done, try to give each tangram piece a buffer (you can use the too_close function to help see which pieces need to be moved). For this, you should start with the outermost pieces, move them apart and then adjust to inner ones.

- You have access to tools like 'move_polygon', 'rotate_polygon', and 'get_observation'.
- **DO NOT** write Python code or markdown blocks.
- **DO NOT** simulate the moves in text.
- **DIRECTLY CALL** the functions to move the pieces.
- Execute one or two moves, then observe, then move again.

get_observation allows you to view the board as an image
get_observation_points tells you the vertices of all shapes
intersections tells you whether your pieces are colliding with each other (which is not allowed)
too_close tells you which pieces are too close to each other (within 10 pixels). This is more optional to follow, but recommended. I suggest that you don't use it until after you're done with your design!

actively make moves. don't get stuck thinking too hard when you can try things and see what works, physically. it's a puzzle you have to interact with!

When finished, just output "done" without any other punctuation.

DO NOT REFER TO ANY EXTERNAL SOURCES, OR MAKE NEW FILES.
            """

            print("sending response")

            response = chat.send_message(f"{system_prompt}\n\nTask: {user_instruction}")

            print("response received")

            while True:
                if response.function_calls:
                    for call in response.function_calls:
                        print(f"Gemini calling: {call.name} with {call.args}")

                        result = await session.call_tool(call.name, arguments=call.args)

                        response = chat.send_message(
                            types.Part.from_function_response(
                                name=call.name,
                                response={"result": result.content}
                            )
                        )
                else:
                    candidate = response.candidates[0]
                    if response.text is None:
                        print("\n!!! RESPONSE BLOCKED OR EMPTY !!!")
                        print(f"Finish Reason: {candidate.finish_reason}")
                        print(f"Safety Ratings: {candidate.safety_ratings}")

                        response = await chat.send_message(
                            "The previous response was blocked. Please try again, but be more concise and avoid harmful keywords."
                        )
                        continue

                    print("Gemini Response:", response.text)
                    if response.text and ("done" in response.text.lower() or "finished" in response.text.lower()):
                        break

                    response = chat.send_message("Continue or confirm completion.")

            final_state_result = await session.call_tool("get_observation_points")
            final_blueprint = json.loads(final_state_result.content[0].text)

            return final_blueprint

def main():
    instruction = "Please arrange these shapes into your best approximation of a house."
    blueprint = asyncio.run(run_planning_cycle(instruction))

    print(json.dumps(blueprint, indent=2))

    # TODO: Publish `blueprint` to ROS2 topic here
    # publisher.publish(json.dumps(blueprint))

if __name__ == "__main__":
    main()