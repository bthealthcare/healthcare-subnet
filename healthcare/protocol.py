# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 demon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
import bittensor as bt
import torch
import pydantic

class Request(bt.Synapse):
    """
    This protocol helps in handling Request request and response communication between
    the miner and the validator.

    Attributes:
    - hf_link: A link of the huggingface model. # username/model_type
    - key: A string value to decrypt the model.
    """

    # Required request output, filled by recieving axon.
    hf_link: str = ""
    version: str = ""
    token: str = ""

    def deserialize(self) -> List[str]:
        """
        Deserialize the output. This method retrieves the response from
        the miner in the form of output_text, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        >>> List[str]: The deserialized response, which in this case is the value of [hf_link, token].
        """
        
        return [self.hf_link, self.token]
