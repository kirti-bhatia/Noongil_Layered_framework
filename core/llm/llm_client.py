"""
====================================================================
NOONGIL-X
LLM CLIENT
====================================================================

Layer:
Core Intelligence Infrastructure

Purpose:
Central interface for Large Language Models.

Primary Backend:
    Ollama (Local LLM)

Future Supported:
    OpenAI
    Claude
    Gemini
    Other APIs

Responsibilities:
    - LLM configuration
    - Provider management
    - Ollama connection
    - Model verification
    - Health monitoring
    - Logging

====================================================================
"""


# ==============================================================
# IMPORTS
# ==============================================================

from __future__ import annotations


import os
import json
import time
import logging
from pathlib import Path
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Generator
)


import requests



# ==============================================================
# LOGGER CONFIGURATION
# ==============================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)


logger = logging.getLogger(
    "NOONGIL.LLM"
)



# ==============================================================
# LLM CONFIGURATION
# ==============================================================


class LLMConfig:
    """
    Central configuration object.

    This avoids hardcoding values
    inside reasoning modules.

    Example:

        config = LLMConfig(
            model="llama3"
        )

    """


    def __init__(
        self,

        provider: str = "ollama",

        model: str = "llama3",

        host: str = "http://localhost:11434",

        timeout: int = 120,

        retry_count: int = 3,

        temperature: float = 0.2,

        max_tokens: int = 2048,

        stream: bool = False

    ):


        self.provider = provider.lower()

        self.model = model

        self.host = host.rstrip("/")

        self.timeout = timeout

        self.retry_count = retry_count

        self.temperature = temperature

        self.max_tokens = max_tokens

        self.stream = stream



    def to_dict(self) -> Dict[str,Any]:

        return {

            "provider":
                self.provider,

            "model":
                self.model,

            "host":
                self.host,

            "timeout":
                self.timeout,

            "retry_count":
                self.retry_count,

            "temperature":
                self.temperature,

            "max_tokens":
                self.max_tokens,

            "stream":
                self.stream

        }



    def __repr__(self):

        return (
            f"LLMConfig("
            f"provider={self.provider}, "
            f"model={self.model})"
        )





# ==============================================================
# CONFIGURATION LOADER
# ==============================================================


class ConfigLoader:
    """
    Loads configuration from environment
    variables.

    Later can support:
        llm_config.json
        yaml
        database

    """


    @staticmethod
    def load() -> LLMConfig:


        return LLMConfig(

            provider=os.getenv(
                "NOONGIL_LLM_PROVIDER",
                "ollama"
            ),


            model=os.getenv(
                "NOONGIL_LLM_MODEL",
                "llama3"
            ),


            host=os.getenv(
                "NOONGIL_LLM_HOST",
                "http://localhost:11434"
            ),


            timeout=int(
                os.getenv(
                    "NOONGIL_LLM_TIMEOUT",
                    "120"
                )
            )

        )





# ==============================================================
# OLLAMA CONNECTION MANAGER
# ==============================================================


class OllamaManager:
    """
    Handles all Ollama communication.

    Responsibilities:

        - Server health
        - Model listing
        - Model availability
        - Connection validation

    """


    def __init__(
        self,
        config: LLMConfig
    ):


        self.config = config

        self.host = config.host


        self.session = requests.Session()



    # ----------------------------------------------------------

    def health_check(self) -> bool:
        """
        Check Ollama server availability.
        """


        try:


            response = self.session.get(

                self.host,

                timeout=5

            )


            if response.status_code == 200:

                logger.info(
                    "Ollama server available"
                )

                return True



        except Exception as error:


            logger.error(
                f"Ollama health failure: {error}"
            )


        return False




    # ----------------------------------------------------------

    def list_models(self) -> List[str]:
        """
        Return installed Ollama models.
        """


        try:


            response = self.session.get(

                f"{self.host}/api/tags",

                timeout=10

            )


            response.raise_for_status()


            data = response.json()



            models = []


            for item in data.get(
                "models",
                []
            ):


                models.append(
                    item.get("name")
                )



            return models



        except Exception as error:


            logger.error(
                f"Unable to fetch models: {error}"
            )


            return []




    # ----------------------------------------------------------

    def model_exists(
        self,
        model_name: Optional[str]=None
    ) -> bool:


        model_name = (
            model_name
            or
            self.config.model
        )


        available_models = (
            self.list_models()
        )


        if model_name in available_models:


            logger.info(
                f"Model found: {model_name}"
            )


            return True



        logger.warning(
            f"Model missing: {model_name}"
        )


        return False




    # ----------------------------------------------------------

    def pull_model(
        self,
        model_name: Optional[str]=None
    ) -> bool:
        """
        Download model if unavailable.

        Used optionally.
        """

        model_name = (
            model_name
            or
            self.config.model
        )


        try:


            response = self.session.post(

                f"{self.host}/api/pull",

                json={
                    "name":
                    model_name
                },

                timeout=None

            )


            return (
                response.status_code
                ==
                200
            )



        except Exception as error:


            logger.error(
                f"Model pull failed: {error}"
            )


            return False





# ==============================================================
# MAIN LLM CLIENT INITIALIZATION
# ==============================================================


class LLMClient:
    """
    Main NOONGIL-X LLM interface.

    All Layer 4 modules should communicate
    through this class.

    Example:

        llm = LLMClient()

        result = llm.generate(
            "Analyze situation"
        )

    """



    def __init__(
        self,
        config: Optional[LLMConfig]=None
    ):


        self.config = (
            config
            or
            ConfigLoader.load()
        )


        logger.info(
            f"Starting LLM Client "
            f"{self.config}"
        )



        if self.config.provider == "ollama":


            self.backend = OllamaManager(
                self.config
            )


        else:


            raise ValueError(
                f"Unsupported provider: "
                f"{self.config.provider}"
            )



        self.history = []



        self._initialize()




    # ----------------------------------------------------------

    def _initialize(self):

        """
        Startup checks.
        """


        if not self.backend.health_check():


            logger.warning(
                "LLM backend unavailable"
            )


        else:


            logger.info(
                "LLM backend initialized"
            )



        if not self.backend.model_exists():


            logger.warning(
                f"Required model "
                f"{self.config.model}"
                " not installed"
            )

    # ==========================================================
    # TEXT GENERATION
    # ==========================================================

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Single prompt generation.

        Used by:
            - Explanation Engine
            - Situation Understanding
            - Reasoning modules

        """


        payload = {

            "model":
                self.config.model,


            "prompt":
                prompt,


            "stream":
                False,


            "options":
            {

                "temperature":
                    temperature
                    if temperature is not None
                    else self.config.temperature,


                "num_predict":
                    self.config.max_tokens

            }

        }



        if system_prompt:


            payload["system"] = system_prompt



        return self._ollama_generate(
            payload
        )





    # ==========================================================
    # CHAT GENERATION
    # ==========================================================


    def chat(
        self,
        messages: List[Dict[str,str]],
        temperature: Optional[float]=None
    ) -> str:
        """
        Conversational generation.

        Message format:

        [
          {
            "role":"system",
            "content":"You are assistant"
          },

          {
            "role":"user",
            "content":"Analyze scene"
          }
        ]

        """


        payload = {

            "model":
                self.config.model,


            "messages":
                messages,


            "stream":
                False,


            "options":
            {

                "temperature":
                    temperature
                    or self.config.temperature

            }

        }



        return self._ollama_chat(
            payload
        )





    # ==========================================================
    # STREAMING GENERATION
    # ==========================================================


    def stream(
        self,
        prompt: str
    ) -> Generator[str,None,None]:
        """
        Token streaming.

        Useful for:
            - Voice assistant response
            - Real-time interaction

        """


        payload = {


            "model":
                self.config.model,


            "prompt":
                prompt,


            "stream":
                True

        }



        try:


            response = self.backend.session.post(

                f"{self.config.host}/api/generate",

                json=payload,

                stream=True,

                timeout=self.config.timeout

            )



            response.raise_for_status()



            for line in response.iter_lines():


                if line:


                    data = json.loads(
                        line.decode("utf-8")
                    )


                    token = data.get(
                        "response",
                        ""
                    )


                    if token:


                        yield token




        except Exception as error:


            logger.error(
                f"Streaming failed: {error}"
            )





    # ==========================================================
    # JSON GENERATION
    # ==========================================================


    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str,Any]]=None
    ) -> Dict[str,Any]:
        """
        Generate machine-readable JSON.

        Used heavily by NOONGIL reasoning.

        Example:

        {
          "risk":"high",
          "reason":"vehicle detected"
        }

        """


        json_prompt = f"""

You are a reasoning engine.

Return ONLY valid JSON.

No markdown.
No explanation.

"""


        if schema:


            json_prompt += f"""

Follow this JSON structure:

{json.dumps(schema,indent=2)}

"""



        json_prompt += f"""

Input:

{prompt}

"""



        response = self.generate(
            json_prompt
        )



        return self._parse_json(
            response
        )





    # ==========================================================
    # STRUCTURED OUTPUT
    # ==========================================================


    def structured_output(
        self,
        prompt: str,
        output_schema: Dict[str,Any]
    ) -> Dict[str,Any]:
        """
        Forces model output according
        to provided schema.

        Example:

        schema = {

            "intent":"",
            "confidence":0.0

        }

        """


        result = self.generate_json(

            prompt,

            schema=output_schema

        )



        validation = self.validate_structure(

            result,

            output_schema

        )



        if not validation:


            logger.warning(
                "Structured output validation failed"
            )



        return result





    # ==========================================================
    # INTERNAL OLLAMA GENERATOR
    # ==========================================================


    def _ollama_generate(
        self,
        payload: Dict[str,Any]
    ) -> str:
        """
        Low level Ollama generate call.
        Retry enabled.
        """


        for attempt in range(

            self.config.retry_count

        ):


            try:


                response = self.backend.session.post(

                    f"{self.config.host}/api/generate",

                    json=payload,

                    timeout=self.config.timeout

                )


                response.raise_for_status()



                data = response.json()



                return data.get(
                    "response",
                    ""
                )



            except Exception as error:



                logger.warning(

                    f"Generation attempt "
                    f"{attempt+1} failed: "
                    f"{error}"

                )


                time.sleep(2)




        raise RuntimeError(
            "LLM generation failed"
        )





    # ==========================================================
    # INTERNAL CHAT CALL
    # ==========================================================


    def _ollama_chat(
        self,
        payload: Dict[str,Any]
    ) -> str:


        for attempt in range(

            self.config.retry_count

        ):


            try:


                response = self.backend.session.post(

                    f"{self.config.host}/api/chat",

                    json=payload,

                    timeout=self.config.timeout

                )



                response.raise_for_status()



                data=response.json()



                return (

                    data
                    .get("message", {})
                    .get("content","")

                )



            except Exception as error:


                logger.warning(

                    f"Chat attempt "
                    f"{attempt+1} failed: "
                    f"{error}"

                )


                time.sleep(2)



        raise RuntimeError(
            "Chat generation failed"
        )





    # ==========================================================
    # JSON PARSER
    # ==========================================================


    def _parse_json(
        self,
        response:str
    )->Dict[str,Any]:

        """
        Safely extract JSON.
        """

        try:

            return json.loads(
                response
            )


        except json.JSONDecodeError:


            logger.error(
                "Invalid JSON received"
            )


            return {

                "error":
                    "invalid_json",


                "raw_response":
                    response

            }





    # ==========================================================
    # STRUCTURE VALIDATOR
    # ==========================================================


    def validate_structure(
        self,
        data:Dict[str,Any],
        schema:Dict[str,Any]
    )->bool:
        """
        Basic schema validation.

        Checks required keys.
        """


        if not isinstance(
            data,
            dict
        ):

            return False



        for key in schema.keys():


            if key not in data:

                return False



        return True

    # ==========================================================
    # RESPONSE VALIDATION
    # ==========================================================

    def validate_response(
        self,
        response: Any,
        expected_type: type = str
    ) -> bool:
        """
        Validate basic LLM response.
        """

        if response is None:
            return False


        if not isinstance(
            response,
            expected_type
        ):
            logger.warning(
                "Invalid response type"
            )
            return False


        if isinstance(response, str):

            if len(response.strip()) == 0:
                return False


        return True



    # ==========================================================
    # JSON VALIDATION
    # ==========================================================

    def validate_json_response(
        self,
        data: Dict[str, Any],
        required_fields: List[str]
    ) -> bool:
        """
        Validate JSON output
        from reasoning modules.
        """


        if not isinstance(data, dict):

            return False



        for field in required_fields:

            if field not in data:

                logger.warning(
                    f"Missing field: {field}"
                )

                return False


        return True



    # ==========================================================
    # PROMPT TEMPLATE ENGINE
    # ==========================================================

    def create_prompt(
        self,
        template: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Dynamic prompt generation.

        Example:

        template =
        '''
        Scene:
        {scene}

        Goal:
        {goal}
        '''

        """


        try:

            return template.format(
                **variables
            )


        except KeyError as error:

            logger.error(
                f"Prompt variable missing: {error}"
            )

            raise



    # ==========================================================
    # MEMORY MANAGEMENT
    # ==========================================================

    def add_memory(
        self,
        role: str,
        content: str
    ):
        """
        Store conversation context.
        """


        self.history.append(

            {
                "role": role,
                "content": content
            }

        )



    def get_memory(
        self
    ) -> List[Dict[str,str]]:

        return self.history



    def clear_memory(self):

        self.history = []



    def chat_with_memory(
        self,
        user_message: str
    ) -> str:
        """
        Chat with previous context.
        """


        self.add_memory(
            "user",
            user_message
        )


        response = self.chat(
            self.history
        )


        self.add_memory(
            "assistant",
            response
        )


        return response



    # ==========================================================
    # DIAGNOSTICS
    # ==========================================================

    def diagnostics(
        self
    ) -> Dict[str,Any]:
        """
        Returns current LLM status.
        """


        return {

            "provider":
                self.config.provider,


            "model":
                self.config.model,


            "ollama_status":
                self.backend.health_check(),


            "model_available":
                self.backend.model_exists(),


            "memory_size":
                len(self.history)

        }



    # ==========================================================
    # MODEL SWITCHING
    # ==========================================================

    def switch_model(
        self,
        model_name: str
    ):
        """
        Dynamically change model.

        Example:

        llama3 -> mistral
        """


        logger.info(
            f"Changing model "
            f"{self.config.model} "
            f"to {model_name}"
        )


        self.config.model = model_name

# ==============================================================
# TESTING SECTION
# ==============================================================


if __name__ == "__main__":


    print("="*60)

    print(
        "NOONGIL-X LLM CLIENT TEST"
    )

    print("="*60)



    llm = LLMClient()



    # -------------------------
    # Diagnostics
    # -------------------------

    print("\nSYSTEM STATUS")

    print(

        json.dumps(
            llm.diagnostics(),
            indent=4
        )

    )



    # -------------------------
    # Normal Generation
    # -------------------------

    print(
        "\nTEXT GENERATION"
    )


    answer = llm.generate(

        """
        Explain how AI can assist
        visually impaired users.
        """

    )


    print(answer)



    # -------------------------
    # JSON Reasoning
    # -------------------------

    print(
        "\nJSON OUTPUT"
    )


    result = llm.generate_json(

        """
        Situation:

        User is walking near road.
        Vehicle approaching.

        Identify danger.
        """

    )


    print(

        json.dumps(
            result,
            indent=4
        )

    )



    # -------------------------
    # Structured Reasoning
    # -------------------------

    print(
        "\nSTRUCTURED OUTPUT"
    )


    decision = llm.structured_output(

        """
        User is blind.

        Obstacle detected.

        Decide safest action.
        """,

        {

            "action":"",
            "reason":"",
            "confidence":0.0

        }

    )


    print(

        json.dumps(
            decision,
            indent=4
        )

    )