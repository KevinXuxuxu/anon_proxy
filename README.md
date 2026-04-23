# Anon Proxy

This project use [OpenAI Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/) ([haggingface version](https://huggingface.co/openai/privacy-filter)) as primary PII identification method, paired with persistant dictionary to achieve consistent reversable PII masking, and provide automatic mask/unmask while proxying openai complied LLM API.

The PII identification part runs the OpenAI Privacy Filter model locally, keeping PII strictly on the device while leveraging frontier LLM capability.