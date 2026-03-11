using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using OllamaSharp.Models.Chat;
using OpenAI.Chat;

namespace SK_DEV
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            var builder = Kernel.CreateBuilder();

            builder.AddOllamaChatCompletion(
                modelId: "llama3", //llama3, deepseek-coder
                endpoint: new Uri("http://localhost:11434"));

            //builder.AddOpenAIChatCompletion(
            //    modelId: "llama3", //llama3, deepseek-coder
            //    apiKey: "not-needed",
            //    endpoint: new Uri("http://localhost:11434/v1")
            //);

            var kernel = builder.Build();

            var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

            var chatSystemPrompt = "You are a friendly AI assistant that answers in a friendly manner"; //Be very brief and direct in your answer
            var maxTokens = 200;
            var temperature = 0.9;

            var settings = new OpenAIPromptExecutionSettings
            {
                ChatSystemPrompt = chatSystemPrompt,
                Temperature = temperature,
                MaxTokens = maxTokens,
            };

            var chatHistory = new ChatHistory();

            //var reducer = new ChatHistoryTruncationReducer(targetCount: 2);
            var reducer = new ChatHistorySummarizationReducer(
                service: chatCompletionService,
                targetCount: 2,
                thresholdCount: 2);

            while (true)
            {
                Console.WriteLine("Enter a prompt (or 'exit' to quit):");

                var prompt = Console.ReadLine();

                if (prompt.Equals("exit", StringComparison.CurrentCultureIgnoreCase))
                {
                    break;
                }

                chatHistory.AddUserMessage(prompt);

                var fullMessage = string.Empty;
                var inputTokenCount = 0L;
                var outputTokenCount = 0L;
                var totalTokenCount = 0L;

                var response = chatCompletionService.GetStreamingChatMessageContentsAsync(
                    chatHistory: chatHistory,
                    executionSettings: settings);

                await foreach (var chunk in response)
                {
                    Console.Write(chunk.Content);
                    fullMessage += chunk.Content;

                    //Check for metdata in chunk
                    if (chunk.Metadata?.TryGetValue("Usage", out var usageObj) == true)
                    {
                        if (usageObj is Microsoft.Extensions.AI.UsageContent usage)
                        {
                            inputTokenCount += usage?.Details?.InputTokenCount ?? 0;
                            outputTokenCount += usage?.Details?.OutputTokenCount ?? 0;
                            totalTokenCount += usage?.Details?.TotalTokenCount ?? 0;
                        }
                    }
                }

                //var response = await chatCompletionService.GetChatMessageContentAsync(
                //    chatHistory: chatHistory,
                //    executionSettings: settings);

                chatHistory.AddAssistantMessage(fullMessage);

                Console.WriteLine($" (Input tokens used: {inputTokenCount})");
                Console.WriteLine($" (Output tokens used: {outputTokenCount})");
                Console.WriteLine($" (Total tokens used: {totalTokenCount})");

                //Console.WriteLine("Response:");
                //Console.WriteLine(response);

                //var chatDone = (ChatDoneResponseStream)response.InnerContent;
                //Console.WriteLine($" (Input tokens used: {chatDone.PromptEvalCount})");
                //Console.WriteLine($" (Output tokens used: {chatDone.EvalCount})");
                //Console.WriteLine($" (Total tokens used: {chatDone.PromptEvalCount + chatDone.EvalCount})");

                var reducedChatHistory = await reducer.ReduceAsync(chatHistory);
                chatHistory = reducedChatHistory != null
                    ? [.. reducedChatHistory]
                    : chatHistory;
            }
        }
    }
}
