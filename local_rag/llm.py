'''
LLM integrations.
'''

from typing import Protocol

from ollama import Client, ResponseError


# LLM Exceptions
class ModelNotProvided(Exception):
    '''
    LLM Model not provided
    '''

    def __init__(self):
        super().__init__('An LLM model was not provided.')

    def __str__(self):
        return 'An LLM model was not provided.'


class ModelNotFound(Exception):
    '''
    LLM Model not found
    '''

    def __init__(self, model: str):
        self.model = model
        super().__init__(f'LLM model "{self.model}" was not found.')

    def __str__(self):
        return f'LLM model not found: {self.model}'


class LLM(Protocol):
    '''
    Base LLM class
    '''

    def ask(self, prompt: str) -> str:
        '''
        Pass a prompt to the LLM and receive and answer.

        :param prompt: The prompt to be passed to the LLM
        :type prompt: str
        :return: Answer from the LLM
        :rtype: str
        '''
        ...


# Maps keys to LLM subclasses
registry: dict[str, LLM] = {}


def register(key: str):
    '''
    Decorator that registers an LLM subclass under a given key.

    :param key: A unique key for LLM subclasses
    :type key: str
    '''
    def decorator(cls):
        registry[key] = cls

    return decorator


def llm_factory(key: str, *args, **kwargs) -> LLM:
    '''
    Factory function that returns the `LLM` instance
    registered under the provided key.

    :param key: Key under which the `LLM` instance is registered.
    (e.g. `'ollama'`)
    :type key: str
    :return: `LLM` instance registered under the provided `key`.
    :rtype: LLM
    '''
    if key not in registry:
        raise KeyError(f'"{key}" is not registered as an LLM provider')

    return registry[key](*args, **kwargs)


@register('ollama')
class OllamaLLM(LLM):
    '''
    Integration for LLMs served by Ollama.
    '''

    def __init__(self, **kwargs):
        host = kwargs.get('host', 'http://localhost:11434')
        self.client = Client(host)
        self.model = kwargs.get('model')

        if not self.model:
            raise ModelNotProvided

    def ask(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]

        try:
            response = self.client.chat(model=self.model, messages=messages)
        except ResponseError as e:
            if e.status_code == 404:
                raise ModelNotFound(self.model) from e
            raise e

        return response['message']['content']
