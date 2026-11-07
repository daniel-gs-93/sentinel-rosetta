"""Pure logic: the example sentence pair shared by the integration tests.

Both sentences are shortened from record f695e433... (line 5281 of
data/llm_generated/en_to_zh/llm_generated.jsonl), whose English sentence
appears in the 2022 U.S. Department of Defense report "Military and Security
Developments Involving the People's Republic of China" (pdf page 92). The
Chinese sentence keeps the services clause that the English sentence drops, so
the two are not translations of each other. The simplified text is the expected
output of the Traditional to Simplified conversion.
"""

from typing import Final

EXAMPLE_ENGLISH: Final[str] = (
    "Their commercial space sector features partially state-owned enterprises."
)
EXAMPLE_CHINESE: Final[str] = (
    "其商業太空部門，包括遙感探測、發射與通信服務，具有部分國有的企業"
)
EXAMPLE_CHINESE_SIMPLIFIED: Final[str] = (
    "其商业太空部门，包括遥感探测、发射与通信服务，具有部分国有的企业"
)
