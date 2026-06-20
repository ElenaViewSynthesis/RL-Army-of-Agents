export async function runAgentLoop(model, messages, { maxIterations = 10 } = {}) {
  let iteration = 0;

  while (true) {
    iteration++;
    if (iteration > maxIterations) {
      throw new Error(
        `Agentic loop exceeded ${maxIterations} iterations without reaching a stop condition. ` +
        `Last finish_reason was 'tool_calls' — possible runaway tool-call cycle.`
      );
    }

    const response     = await model.predict(messages);
    const message      = response.choices[0].message;
    const finishReason = response.choices[0].finishReason;
    messages.push(message);

    if (finishReason === 'stop' || finishReason === 'end_turn') {
      console.error(`\nReport complete.\n`);
      return message.content || '';
    }

    if (finishReason === 'tool_calls') {
      const toolCalls = message.toolCalls || [];
      console.error(`[Step ${iteration}] Fetching data via ${toolCalls.length} tool(s):`);
      toolCalls.forEach((tc) => console.error(`  → ${tc.function.name}(${tc.function.arguments})`));

      const toolResults = await Promise.all(
        toolCalls.map(async (tc) => {
          try {
            const input   = JSON.parse(tc.function.arguments || '{}');
            const result  = await model.callTool(tc.function.name, input);
            const content = JSON.stringify(result);
            console.error(`  ✓ ${tc.function.name} — ${content.length} chars`);
            return { role: 'tool', toolCallId: tc.id, content };
          } catch (err) {
            console.error(`  ✗ ${tc.function.name} — ${err.message}`);
            return { role: 'tool', toolCallId: tc.id, content: `Error retrieving data: ${err.message}` };
          }
        })
      );

      messages.push(...toolResults);
    }
  }
}
