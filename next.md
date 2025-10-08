# Next.md

So now we are going to make it a bit more fun. Locally (but we are on wsl2 so 127.0.0.1 doesnt really work, please help me resolve it) we have this super fast 3b param model:

curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-4b-2507",
    "messages": [
      { "role": "system", "content": "Always answer in rhymes. Today is Thursday" },
      { "role": "user", "content": "What day is it today?" }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
}'


Now, what we want to do is to use this model on a new iteration of the client app - 
when we get new text from the user (message A), we want to send it to this local model - only the text
and then we want to send the response to a local tts endpoint that sends back audio. 
if the user says more stuff (and the vad is activated) before we have started to play the audio, we abort any calls, wait for that vad to finish and to be transcribed (message B) and send message C = message A + message B to the llm, followed by the encoding.

Remember, we are on wsl2 ubuntu, so playing the sound has do be done in a way that makes sense.

If we have started to play the sound from the ai response, we fade it out and treat it as if it never happened.