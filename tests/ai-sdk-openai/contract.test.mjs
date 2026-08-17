import assert from 'node:assert/strict';
import test from 'node:test';

import { createOpenAI } from '@ai-sdk/openai';
import { generateText, jsonSchema, streamText, tool } from 'ai';

const env = {
  baseURL: process.env.NINFER_BASE_URL,
  model: process.env.NINFER_MODEL,
};
const liveSkip = env.baseURL && env.model
  ? false
  : 'set NINFER_BASE_URL and NINFER_MODEL to run against a live server';

function apiBaseURL(value) {
  const base = value.replace(/\/+$/, '');
  // ninfer-serve is commonly given as an origin, while createOpenAI expects the
  // API prefix. Accepting either form prevents accidental /v1/v1 requests.
  return base.endsWith('/v1') ? base : `${base}/v1`;
}

function captureAndForward(requests, fetchImpl = globalThis.fetch) {
  return async (input, init = {}) => {
    const request = input instanceof Request ? input : null;
    const rawBody = init.body ?? (request ? await request.clone().text() : undefined);
    requests.push({
      url: String(request?.url ?? input),
      method: init.method ?? request?.method ?? 'GET',
      body: rawBody ? JSON.parse(rawBody) : undefined,
    });
    return fetchImpl(input, init);
  };
}

function provider(baseURL, fetch) {
  return createOpenAI({
    baseURL: apiBaseURL(baseURL),
    apiKey: 'ninfer-contract',
    fetch,
  });
}

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });
}

function chatResponse(text = 'mock answer') {
  return {
    id: 'chatcmpl_contract',
    object: 'chat.completion',
    created: 1,
    model: 'contract-model',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text },
      finish_reason: 'stop',
      logprobs: null,
    }],
    usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
  };
}

function responsesResponse({
  reasoning = false,
  model = 'contract-model',
  store = true,
} = {}) {
  return {
    id: 'resp_contract',
    object: 'response',
    created_at: 1,
    model,
    store,
    status: 'completed',
    output: [
      ...(reasoning ? [{
        id: 'rs_contract',
        type: 'reasoning',
        summary: [{ type: 'summary_text', text: 'mock reasoning' }],
      }] : []),
      {
        id: 'msg_contract',
        type: 'message',
        status: 'completed',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'mock answer', annotations: [] }],
      },
    ],
    usage: {
      input_tokens: 7,
      input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
      output_tokens: 5,
      output_tokens_details: { reasoning_tokens: reasoning ? 2 : 0 },
      total_tokens: 12,
    },
  };
}

function sseResponse(events) {
  const payload = `${events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('')}data: [DONE]\n\n`;
  return new Response(payload, {
    headers: { 'content-type': 'text/event-stream' },
  });
}

function chatStreamResponse() {
  const common = {
    id: 'chatcmpl_stream_contract',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'contract-model',
  };
  return sseResponse([
    { ...common, choices: [{ index: 0, delta: { role: 'assistant', content: 'mock ' }, finish_reason: null }] },
    { ...common, choices: [{ index: 0, delta: { content: 'stream' }, finish_reason: null }] },
    {
      ...common,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    },
    { ...common, choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
  ]);
}

function responsesStreamResponse() {
  const terminal = responsesResponse();
  terminal.output[0].content[0].text = 'mock stream';
  const events = [
    { type: 'response.created', response: { ...terminal, status: 'in_progress', output: [], usage: null } },
    { type: 'response.in_progress', response: { ...terminal, status: 'in_progress', output: [], usage: null } },
    {
      type: 'response.output_item.added',
      output_index: 0,
      item: { id: 'msg_contract', type: 'message', status: 'in_progress', role: 'assistant', content: [] },
    },
    {
      type: 'response.content_part.added',
      item_id: 'msg_contract',
      output_index: 0,
      content_index: 0,
      part: { type: 'output_text', text: '', annotations: [] },
    },
    { type: 'response.output_text.delta', item_id: 'msg_contract', output_index: 0, content_index: 0, delta: 'mock stream' },
    {
      type: 'response.output_text.done',
      item_id: 'msg_contract',
      output_index: 0,
      content_index: 0,
      text: 'mock stream',
    },
    {
      type: 'response.content_part.done',
      item_id: 'msg_contract',
      output_index: 0,
      content_index: 0,
      part: terminal.output[0].content[0],
    },
    { type: 'response.output_item.done', output_index: 0, item: terminal.output[0] },
    { type: 'response.completed', response: terminal },
  ];
  // Responses streams use named events and sequence numbers, but no Chat [DONE] sentinel.
  const payload = events.map((event, sequence_number) => {
    const value = { ...event, sequence_number };
    return `event: ${value.type}\ndata: ${JSON.stringify(value)}\n\n`;
  }).join('');
  return new Response(payload, { headers: { 'content-type': 'text/event-stream' } });
}

const weatherSchema = jsonSchema({
  type: 'object',
  properties: { city: { type: 'string' } },
  required: ['city'],
  additionalProperties: false,
});

test('plain generateText uses Chat Completions and exposes usage', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid', captureAndForward(
    requests,
    async () => jsonResponse(chatResponse()),
  ));

  const result = await generateText({
    model: openai.chat('contract-model'),
    prompt: 'Reply briefly.',
    maxOutputTokens: 32,
  });

  assert.equal(result.text, 'mock answer');
  assert.equal(result.usage.inputTokens, 4);
  assert.equal(result.usage.outputTokens, 2);
  assert.equal(requests.length, 1);
  assert.equal(new URL(requests[0].url).pathname, '/v1/chat/completions');
  assert.equal(requests[0].body.model, 'contract-model');
  assert.equal(requests[0].body.stream, undefined);
  assert.equal(requests[0].body.messages.at(-1).content, 'Reply briefly.');
});

test('plain streamText consumes Chat Completions SSE and captures its body', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid/v1', captureAndForward(
    requests,
    async () => chatStreamResponse(),
  ));

  const result = streamText({
    model: openai.chat('contract-model'),
    prompt: 'Stream briefly.',
    maxOutputTokens: 32,
  });
  let text = '';
  for await (const delta of result.textStream) text += delta;

  assert.equal(text, 'mock stream');
  assert.equal((await result.usage).totalTokens, 5);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].body.stream, true);
  assert.deepEqual(requests[0].body.stream_options, { include_usage: true });
});

test('Responses reasoning options and detailed usage survive the SDK boundary', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid', captureAndForward(
    requests,
    async () => jsonResponse(responsesResponse({ reasoning: true, model: 'custom-ninfer-model' })),
  ));

  const result = await generateText({
    model: openai.responses('custom-ninfer-model'),
    prompt: 'Reason briefly.',
    maxOutputTokens: 32,
    providerOptions: {
      openai: {
        forceReasoning: true,
        reasoningEffort: 'low',
        reasoningSummary: 'detailed',
      },
    },
  });

  assert.equal(result.reasoningText, 'mock reasoning');
  assert.equal(result.usage.inputTokens, 7);
  assert.equal(result.usage.outputTokens, 5);
  assert.equal(result.usage.outputTokenDetails.reasoningTokens, 2);
  assert.equal(result.usage.totalTokens, 12);
  assert.equal(new URL(requests[0].url).pathname, '/v1/responses');
  assert.deepEqual(requests[0].body.reasoning, { effort: 'low', summary: 'detailed' });
});

test('Responses streamText consumes native Responses SSE', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid', captureAndForward(
    requests,
    async () => responsesStreamResponse(),
  ));

  const result = streamText({
    model: openai.responses('contract-model'),
    prompt: 'Stream briefly.',
    maxOutputTokens: 32,
  });
  let text = '';
  for await (const delta of result.textStream) text += delta;

  assert.equal(text, 'mock stream');
  assert.equal((await result.usage).totalTokens, 12);
  assert.equal(requests[0].body.stream, true);
  assert.equal(new URL(requests[0].url).pathname, '/v1/responses');
});

test('Responses serializes deterministic tool continuation history', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid', captureAndForward(
    requests,
    async () => jsonResponse(responsesResponse()),
  ));
  const call = { type: 'tool-call', toolCallId: 'call_weather_1', toolName: 'weather', input: { city: 'Paris' } };

  await generateText({
    model: openai.responses('contract-model'),
    tools: { weather: tool({ description: 'Get weather', inputSchema: weatherSchema }) },
    messages: [
      { role: 'user', content: 'What is the weather in Paris?' },
      { role: 'assistant', content: [call] },
      {
        role: 'tool',
        content: [{
          type: 'tool-result',
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          input: call.input,
          output: { type: 'json', value: { temperatureC: 20 } },
        }],
      },
    ],
    maxOutputTokens: 32,
  });

  const functionCall = requests[0].body.input.find(item => item.type === 'function_call');
  const functionOutput = requests[0].body.input.find(item => item.type === 'function_call_output');
  assert.deepEqual(functionCall, {
    type: 'function_call',
    call_id: 'call_weather_1',
    name: 'weather',
    arguments: '{"city":"Paris"}',
  });
  assert.deepEqual(functionOutput, {
    type: 'function_call_output',
    call_id: 'call_weather_1',
    output: '{"temperatureC":20}',
  });
});

test('Responses exposes public summaries with store:false and the ignored include hint', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid', captureAndForward(
    requests,
    async () => jsonResponse(responsesResponse({ reasoning: true, store: false })),
  ));
  const result = await generateText({
    model: openai.responses('contract-model'),
    prompt: 'Reason publicly.',
    maxOutputTokens: 32,
    providerOptions: { openai: { forceReasoning: true, store: false } },
  });

  // AI SDK asks for encrypted content when store:false. NInfer accepts this as an ignored hint and
  // returns the same public summary_text representation; native schema tests cover replay because
  // this pinned SDK drops unencrypted reasoning when it rebuilds response messages.
  assert.equal(result.reasoningText, 'mock reasoning');
  assert.equal(requests[0].body.store, false);
  assert.deepEqual(requests[0].body.include, ['reasoning.encrypted_content']);
});

test('Responses capability errors remain inspectable AI SDK call errors', async () => {
  const requests = [];
  const openai = provider('http://contract.invalid', captureAndForward(requests, async (_input, init) => {
    const body = JSON.parse(init.body);
    const code = body.tools[0].strict
      ? 'strict_tools_not_supported'
      : 'tool_choice_not_supported';
    return jsonResponse({
      error: {
        message: `unsupported capability: ${code}`,
        type: 'invalid_request_error',
        param: code.startsWith('strict') ? 'tools' : 'tool_choice',
        code,
      },
    }, { status: 400 });
  }));
  const rejectWithCode = async (options, code) => assert.rejects(
    generateText(options),
    error => {
      assert.equal(error.statusCode, 400);
      assert.equal(JSON.parse(error.responseBody).error.code, code);
      return true;
    },
  );
  const common = {
    model: openai.responses('contract-model'),
    prompt: 'Use the weather tool.',
    maxOutputTokens: 32,
  };

  await rejectWithCode({
    ...common,
    tools: { weather: tool({ inputSchema: weatherSchema, strict: true }) },
  }, 'strict_tools_not_supported');
  await rejectWithCode({
    ...common,
    tools: { weather: tool({ inputSchema: weatherSchema }) },
    toolChoice: 'required',
  }, 'tool_choice_not_supported');

  assert.equal(requests[0].body.tools[0].strict, true);
  assert.equal(requests[1].body.tool_choice, 'required');
});

async function expectCapabilityError(options, code) {
  await assert.rejects(
    generateText(options),
    error => {
      assert.equal(error.statusCode, 400);
      assert.equal(JSON.parse(error.responseBody).error.code, code);
      return true;
    },
  );
}

test('live plain generateText and streamText', { skip: liveSkip }, async () => {
  const requests = [];
  const openai = provider(env.baseURL, captureAndForward(requests));

  const generated = await generateText({
    model: openai.chat(env.model),
    prompt: 'Reply with one short sentence.',
    // Qwen may spend a small budget entirely in its reasoning channel, which
    // Chat Completions does not expose through AI SDK's text result.
    maxOutputTokens: 128,
  });
  const streamed = streamText({
    model: openai.chat(env.model),
    prompt: 'Reply with one short sentence.',
    maxOutputTokens: 128,
  });
  let streamedText = '';
  for await (const delta of streamed.textStream) streamedText += delta;

  assert.ok(generated.text.length > 0);
  assert.ok(streamedText.length > 0);
  assert.ok(generated.usage.inputTokens > 0);
  assert.ok((await streamed.usage).inputTokens > 0);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].body.stream, undefined);
  assert.equal(requests[1].body.stream, true);
});

test('live Responses reasoning and usage', { skip: liveSkip }, async () => {
  const requests = [];
  const openai = provider(env.baseURL, captureAndForward(requests));
  const result = await generateText({
    model: openai.responses(env.model),
    prompt: 'State whether 2 + 2 equals 4, briefly.',
    maxOutputTokens: 64,
    providerOptions: {
      openai: {
        forceReasoning: true,
        reasoningEffort: 'low',
        reasoningSummary: 'detailed',
      },
    },
  });

  const reasoningTokens = result.usage.outputTokenDetails.reasoningTokens;
  assert.ok(result.text.length > 0 || (result.reasoningText?.length ?? 0) > 0);
  if (reasoningTokens > 0) assert.ok((result.reasoningText?.length ?? 0) > 0);
  assert.ok(result.usage.inputTokens > 0);
  assert.ok(result.usage.outputTokens > 0);
  assert.ok(reasoningTokens >= 0);
  assert.deepEqual(requests[0].body.reasoning, { effort: 'low', summary: 'detailed' });
});

test('live Responses streamText and detailed usage', { skip: liveSkip }, async () => {
  const requests = [];
  const openai = provider(env.baseURL, captureAndForward(requests));
  const result = streamText({
    model: openai.responses(env.model),
    prompt: 'Reply with one short sentence.',
    maxOutputTokens: 64,
  });
  let text = '';
  for await (const delta of result.textStream) text += delta;
  const usage = await result.usage;

  assert.ok(text.length > 0 || usage.outputTokenDetails.reasoningTokens > 0);
  assert.ok(usage.inputTokens > 0);
  assert.ok(usage.outputTokens > 0);
  assert.equal(requests[0].body.stream, true);
});

test('live Responses supports non-strict serialized tool history', { skip: liveSkip }, async () => {
  const requests = [];
  const openai = provider(env.baseURL, captureAndForward(requests));
  const call = { type: 'tool-call', toolCallId: 'call_weather_live', toolName: 'weather', input: { city: 'Paris' } };
  const result = await generateText({
    model: openai.responses(env.model),
    tools: { weather: tool({ description: 'Get weather', inputSchema: weatherSchema }) },
    messages: [
      { role: 'user', content: 'What is the weather in Paris?' },
      { role: 'assistant', content: [call] },
      {
        role: 'tool',
        content: [{
          type: 'tool-result',
          toolCallId: call.toolCallId,
          toolName: call.toolName,
          input: call.input,
          output: { type: 'json', value: { temperatureC: 20 } },
        }],
      },
      { role: 'user', content: 'State the reported temperature briefly.' },
    ],
    maxOutputTokens: 64,
  });

  assert.ok(result.text.length > 0 || result.usage.outputTokens > 0);
  // Omitted and false are the two wire representations of a non-strict tool.
  assert.equal(requests[0].body.tools[0].strict ?? false, false);
  assert.equal(requests[0].body.input.find(item => item.type === 'function_call').arguments, '{"city":"Paris"}');
  assert.equal(requests[0].body.input.find(item => item.type === 'function_call_output').output, '{"temperatureC":20}');
});

test('live store:false Responses returns public reasoning with the ignored include hint', { skip: liveSkip }, async () => {
  const requests = [];
  const openai = provider(env.baseURL, captureAndForward(requests));
  const result = await generateText({
    model: openai.responses(env.model),
    prompt: 'Decide whether 3 is odd, then answer briefly.',
    maxOutputTokens: 64,
    providerOptions: { openai: { forceReasoning: true, store: false } },
  });

  assert.deepEqual(requests[0].body.include, ['reasoning.encrypted_content']);
  assert.equal(requests[0].body.store, false);
  if (result.usage.outputTokenDetails.reasoningTokens > 0) {
    assert.ok((result.reasoningText?.length ?? 0) > 0);
  }
  assert.ok(result.text.length > 0 || result.usage.outputTokens > 0);
});

test('live Responses reports unsupported capabilities explicitly', { skip: liveSkip }, async () => {
  const requests = [];
  const openai = provider(env.baseURL, captureAndForward(requests));
  const common = {
    model: openai.responses(env.model),
    prompt: 'Use the weather tool.',
    maxOutputTokens: 32,
  };

  await expectCapabilityError({
    ...common,
    tools: { weather: tool({ inputSchema: weatherSchema, strict: true }) },
  }, 'strict_tools_not_supported');
  await expectCapabilityError({
    ...common,
    tools: { weather: tool({ inputSchema: weatherSchema }) },
    toolChoice: 'required',
  }, 'tool_choice_not_supported');

  assert.equal(requests[0].body.tools[0].strict, true);
  assert.equal(requests[1].body.tool_choice, 'required');
});
