import os
import asyncio
from pathlib import Path

# Add references# Add references
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.functions import kernel_function
from typing import Annotated

## Loading the .env file
load_dotenv()

async def main():
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')

    # Load the expnses data file
    script_dir = Path(__file__).parent
    file_path = script_dir / 'data.txt'
    with file_path.open('r') as file:
        data = file.read() + "\n"

    # Ask for a prompt
    user_prompt = input(f"Here is the expenses data in your file:\n\n{data}\n\nWhat would you like me to do with it?\n\n")
    
    # Run the async agent code
    await process_expenses_data (user_prompt, data)
    
async def process_expenses_data(prompt, expenses_data):

    # Get configuration settings
    ## Identity Based login
    ai_agent_settings = AzureAIAgentSettings(
        agent_id=os.environ['AZURE_AGENT_ID'],
        model_deployment_name=os.environ['AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME']
    )
    # Connect to the Azure AI Foundry project
    async with (
        ## Adding creds to login
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True) as creds,
        AzureAIAgent.create_client(
            credential=creds
        ) as project_client,
    ):
        
        # Define an Azure AI agent that sends an expense claim email
        expenses_agent_def = await project_client.agents.create_agent(
            model= ai_agent_settings.model_deployment_name,
            name="expenses_agent",
            instructions="""You are an AI assistant for expense claim submission.
                            When a user submits expenses data and requests an expense claim, use the plug-in function to send an email to skyrayzor1@gmail.com with the subject 'Expense Claim`and a body that contains itemized expenses with a total.
                            Then confirm to the user that you've done so.
                            Send email from: merbiren@gmail.com
                           
                            """
        )


        # Create a semantic kernel agent 
        # (Should be visible in overview > agents in Azure foundry UI)
        expenses_agent = AzureAIAgent(
            client=project_client,
            definition=expenses_agent_def,
            plugins=[]
        )


        # Use the agent to process the expenses data
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread | None = None
        try:
            # Add the input prompt to a list of messages to be submitted
            prompt_messages = [f"{prompt}: {expenses_data}"]
            
            # Invoke the agent for the specified thread with the messages
            response = await expenses_agent.get_response(prompt_messages, thread=thread)
            # Display the response
            print(f"\n# {response.name}:\n{response}")
        except Exception as e:
            # Something went wrong
            print (e)
        finally:
            # Cleanup: Delete the thread and agent
            await thread.delete() if thread else None
            await project_client.agents.delete_agent(expenses_agent.id)



if __name__ == "__main__":
    asyncio.run(main())
