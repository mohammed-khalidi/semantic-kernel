using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureAIInference;
using Microsoft.SemanticKernel.Connectors.HuggingFace;
using Microsoft.SemanticKernel.Connectors.Onnx;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using OllamaSharp.Models.Chat;
using OpenAI.Chat;
using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SK_DEV
{
    internal class Program
    {
        private const string OPENAI_API_KEY = "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";

        private const string GITHUB_ENDPOINT = "https://models.github.ai/inference";
        private const string GITHUB_API_KEY = "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";
        private const string GITHUB_MODEL_ID = "openai/gpt-4.1-nano";

        private const string HUGGING_FACE_ENDPOINT = "https://router.huggingface.co";
        private const string HUGGING_FACE_API_KEY = "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";
        private const string HUGGING_FACE_MODEL_ID = "deepseek-ai/DeepSeek-R1:sambanova";

        private const string ONNX_PATH = "D:\\projects\\source_code\\research\\ai-models\\Phi-4-multimodal-instruct-onnx\\gpu\\gpu-int4-rtn-block-32";
        private const string ONNX_MODEL_ID = "phi-4";

        private const string IMAGE_DIR = "D:\\projects\\source_code\\research\\SK_Course\\module4\\ai_multimodal\\images";

        static async Task Main(string[] args)
        {
            var builder = Kernel.CreateBuilder();

            //builder.AddOllamaChatCompletion(
            //    modelId: "llama3", //llama3, deepseek-coder
            //    endpoint: new Uri("http://localhost:11434"));

            builder.AddOnnxRuntimeGenAIChatCompletion(
                modelId: ONNX_MODEL_ID, //llama3, deepseek-coder
                modelPath: ONNX_PATH);

            //builder.AddAzureAIInferenceChatCompletion(
            //    modelId: GITHUB_MODEL_ID,
            //    apiKey: GITHUB_API_KEY,
            //    endpoint: new Uri(GITHUB_ENDPOINT)
            //);

            ////Hugging Face
            //builder.AddHuggingFaceChatCompletion(
            //    model: HUGGING_FACE_MODEL_ID,
            //    apiKey: HUGGING_FACE_API_KEY,
            //    endpoint: new Uri(HUGGING_FACE_ENDPOINT));

            var kernel = builder.Build();

            var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

            var chatSystemPrompt = @"you are a traffic analyzer AI that monitors traffic congestion images and congestion level.
Heavy congestion level is when there is very little room between cars and vehicles are breaking.
Medium congestion is when there is a lot of cars but they are not braking.
Low traffic is when there are few cars on the road
In addition, attempt to determine if the image was taken with a malfunctioning camera by looking for distorted image";

            var maxTokens = 2000;
            var temperature = 0.9;

            //var settings = new OpenAIPromptExecutionSettings
            //{
            //    ChatSystemPrompt = chatSystemPrompt,
            //    Temperature = temperature,
            //    MaxTokens = maxTokens,
            //};

            //var settings = new AzureAIInferencePromptExecutionSettings
            //{
            //    Temperature = (float)temperature,
            //    MaxTokens = maxTokens,
            //};

            ////Hugging Face prompt execution settings
            //var settings = new HuggingFacePromptExecutionSettings
            //{
            //    Temperature = (float)temperature,
            //    MaxTokens = maxTokens,
            //};

            //Hugging Face prompt execution settings
            var settings = new OnnxRuntimeGenAIPromptExecutionSettings
            {
                Temperature = (float)temperature,
                MaxTokens = maxTokens
            };

            var chatHistory = new ChatHistory(systemMessage: chatSystemPrompt);

            //var reducer = new ChatHistoryTruncationReducer(targetCount: 2);

            //var reducer = new ChatHistorySummarizationReducer(
            //    service: chatCompletionService,
            //    targetCount: 2,
            //    thresholdCount: 2);

            //Get images from directory
            var images = Directory.GetFiles(IMAGE_DIR, "*.jpg");

            for (int i = 0; i < images.Length; i++)
            {
                var fileName = images[i];
                Console.WriteLine($"Adding image to chat history: {fileName}");
                var imageData = File.ReadAllBytes(fileName);

                chatHistory.AddUserMessage(
                    [
                    new ImageContent(imageData, "image/jpeg"),
                    new TextContent($"Analyze the image and determine traffic congestion level. Also determine if camera is malfunctioning.")
                    ]);

                try
                {
                    var response = await chatCompletionService.GetChatMessageContentAsync(
                        chatHistory: chatHistory,
                        executionSettings: settings);

                    Console.WriteLine(response.Content);
                    Console.WriteLine(new string('-', 40));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing image {fileName}: {ex.Message}");
                }

                //Artificial delay
                await Task.Delay(1000);
            }

            //while (true)
            //{
            //    Console.WriteLine("Enter a prompt (or 'exit' to quit):");

            //    var prompt = Console.ReadLine();

            //    if (prompt.Equals("exit", StringComparison.CurrentCultureIgnoreCase))
            //    {
            //        break;
            //    }

            //    chatHistory.AddUserMessage(prompt);

            //    var fullMessage = string.Empty;
            //    var inputTokenCount = 0L;
            //    var outputTokenCount = 0L;
            //    var totalTokenCount = 0L;

            //    var response = chatCompletionService.GetStreamingChatMessageContentsAsync(
            //        chatHistory: new ChatHistory(prompt),
            //        executionSettings: settings);

            //    await foreach (var chunk in response)
            //    {
            //        Console.Write(chunk.Content);
            //        fullMessage += chunk.Content;

            //        //Check for metdata in chunk
            //        if (chunk.Metadata?.TryGetValue("Usage", out var usageObj) == true)
            //        {
            //            if (usageObj is Microsoft.Extensions.AI.UsageContent usage)
            //            {
            //                inputTokenCount += usage?.Details?.InputTokenCount ?? 0;
            //                outputTokenCount += usage?.Details?.OutputTokenCount ?? 0;
            //                totalTokenCount += usage?.Details?.TotalTokenCount ?? 0;
            //            }
            //        }
            //    }

            //    //var response = await chatCompletionService.GetChatMessageContentAsync(
            //    //    chatHistory: chatHistory,
            //    //    executionSettings: settings);

            //    chatHistory.AddAssistantMessage(fullMessage);

            //    Console.WriteLine($" (Input tokens used: {inputTokenCount})");
            //    Console.WriteLine($" (Output tokens used: {outputTokenCount})");
            //    Console.WriteLine($" (Total tokens used: {totalTokenCount})");

            //    //Console.WriteLine("Response:");
            //    //Console.WriteLine(response);

            //    //var chatDone = (ChatDoneResponseStream)response.InnerContent;
            //    //Console.WriteLine($" (Input tokens used: {chatDone.PromptEvalCount})");
            //    //Console.WriteLine($" (Output tokens used: {chatDone.EvalCount})");
            //    //Console.WriteLine($" (Total tokens used: {chatDone.PromptEvalCount + chatDone.EvalCount})");

            //    var reducedChatHistory = await reducer.ReduceAsync(chatHistory);
            //    chatHistory = reducedChatHistory != null
            //        ? [.. reducedChatHistory]
            //        : chatHistory;
            //}

            ((OnnxRuntimeGenAIChatCompletionService)chatCompletionService).Dispose();
        }
    }
}
