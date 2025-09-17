import asyncio
import os
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread

load_dotenv()

# Get configuration settings
ai_agent_settings = AzureAIAgentSettings(
    agent_id=os.environ['AZURE_AGENT_ID'],
    model_deployment_name=os.environ['AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME']
)

async def main() -> None:
    print("Welcome to the chat bot!\n  Type 'exit' to exit.\n  Try to get some billing or refund help.")
    
    # Connect to the Azure AI Foundry project
    async with (
        DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
        ) as creds,
        AzureAIAgent.create_client(credential=creds) as project_client,
    ):
        
        # Create billing agent
        billing_agent_def = await project_client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="BillingAgent",
            instructions="You handle billing issues like charges, payment methods, cycles, fees, discrepancies, and payment failures. Provide detailed and helpful responses to billing-related questions.",
        )
        
        billing_agent = AzureAIAgent(
            client=project_client,
            definition=billing_agent_def,
            plugins=[]
        )
        
        # Create refund agent
        refund_agent_def = await project_client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="RefundAgent",
            instructions="Assist users with refund inquiries, including eligibility, policies, processing, and status updates. Provide clear information about refund processes and requirements.",
        )
        
        refund_agent = AzureAIAgent(
            client=project_client,
            definition=refund_agent_def,
            plugins=[]
        )
        
        # Create triage agent (this will be the main agent users interact with)
        triage_agent_def = await project_client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="TriageAgent",
            instructions="""You are a customer service triage agent. Analyze user requests and determine if they are:
            1. Billing-related issues (charges, payment methods, cycles, fees, discrepancies, payment failures)
            2. Refund-related issues (refund requests, eligibility, policies, processing, status updates)
            
            For billing issues, provide helpful billing assistance.
            For refund issues, provide helpful refund assistance.
            For general inquiries, provide appropriate guidance.
            
            Always be helpful, professional, and provide complete answers to user questions.""",
        )
        
        triage_agent = AzureAIAgent(
            client=project_client,
            definition=triage_agent_def,
            plugins=[]
        )
        
        # Initialize thread for the main conversation
        thread: AzureAIAgentThread | None = None
        
        try:
            while True:
                user_input = input("User:> ")
                
                if user_input.lower().strip() == "exit":
                    print("\n\nExiting chat...")
                    break
                
                # Determine which agent to use based on keywords
                agent_to_use = triage_agent  # Default to triage
                
                # Simple keyword-based routing
                billing_keywords = ['billing', 'charge', 'payment', 'fee', 'bill', 'invoice', 'subscription']
                refund_keywords = ['refund', 'return', 'money back', 'reimburse', 'cancel']
                
                user_lower = user_input.lower()
                
                if any(keyword in user_lower for keyword in billing_keywords):
                    agent_to_use = billing_agent
                    print("(Routing to Billing Agent)")
                elif any(keyword in user_lower for keyword in refund_keywords):
                    agent_to_use = refund_agent
                    print("(Routing to Refund Agent)")
                else:
                    print("(Using Triage Agent)")
                
                # Get response from the selected agent
                response = await agent_to_use.get_response([user_input], thread=thread)
                
                if response:
                    print(f"Agent :> {response}")
                else:
                    print("Agent :> I'm sorry, I couldn't process your request. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nChat interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Cleanup: Delete the thread and agents
            if thread:
                await thread.delete()
            
            # Clean up agents
            try:
                await project_client.agents.delete_agent(billing_agent.id)
                await project_client.agents.delete_agent(refund_agent.id)  
                await project_client.agents.delete_agent(triage_agent.id)
                print("Agents cleaned up successfully.")
            except Exception as e:
                print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    asyncio.run(main())

#### Example output
"""
Welcome to the chat bot!
  Type 'exit' to exit.
  Try to get some billing or refund help.
User:> hi
(Using Triage Agent)
Agent :> Hello! How can I assist you today? If you have any billing or refund-related questions, feel free to let me know.
User:> Need to inquire about an item I just returned
(Routing to Refund Agent)
Agent :> I can help with that! Could you please provide the order number or the item details and the date you returned it? This will help me check the status of your return.
User:> Order number 12121
(Using Triage Agent)
Agent :> Thank you for providing your order number 12121. Could you please specify if your inquiry is related to a billing issue, a refund request, or a general question about your order? This will help me assist you more effectively.
User:> exit


Exiting chat...
Agents cleaned up successfully.
"""