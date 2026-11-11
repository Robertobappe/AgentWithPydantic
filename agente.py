"""
Agente nível 2 — o mesmo agente do nível 1, agora com o framework Pydantic AI.

O QUE MUDA EM RELAÇÃO AO NÍVEL 1
    No nível 1 nós escrevemos à mão: o POST para a API, o JSON Schema de cada
    ferramenta, o loop "modelo pede → código executa", o limite de passos e o
    tratamento de erros. Aqui o framework faz tudo isso. Sobra para nós:

      1. criar o agente (modelo + instruções);
      2. escrever as ferramentas como funções Python com type hints e docstring;
      3. chamar agente.run_sync(pergunta).

    | Nível 1 (na mão)                        | Nível 2 (Pydantic AI)                          |
    |-----------------------------------------|------------------------------------------------|
    | chamar_modelo() com urllib              | o framework chama a API                        |
    | ESQUEMA_FERRAMENTAS escrito à mão       | gerado dos type hints + docstring              |
    | dicionário FERRAMENTAS                  | decorador @agente.tool_plain                   |
    | rodar_agente() com for + MAX_PASSOS     | run_sync() + UsageLimits(request_limit=8)      |
    | "ERRO ..." devolvido como texto         | ModelRetry: o modelo recebe o erro e tenta de novo |
    | validar data com fromisoformat          | type hint `date`: o Pydantic valida sozinho    |
    | lista de dicts e rollback manual        | message_history; falha não altera o histórico  |

COMO O ARQUIVO ESTÁ ORGANIZADO
    1. Configuração   → modelo, chave e limites
    2. Agente         → criação do agente com as instruções
    3. Ferramentas    → funções registradas com @agente.tool_plain
    4. Terminal       → chat interativo (main)

Instalação:
    pip install -r requirements.txt

Uso:
    export GROQ_API_KEY="gsk_..."
    python agente.py

Trocar de modelo (formato "provedor:modelo"):
    MODELO="groq:llama-3.3-70b-versatile" python agente.py
"""

import ast
import json
import operator
import os
import urllib.parse
import urllib.request
from datetime import date, datetime

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.usage import UsageLimits

# ---------------------------------------------------------------------------
# 1. Configuração
# ---------------------------------------------------------------------------
# O modelo é uma string "provedor:modelo". O framework sabe qual API chamar,
# qual URL usar e qual variável de ambiente tem a chave. No nível 1 isso era o
# dicionário PROVEDORES inteiro.

MODELO = os.environ.get("MODELO", "groq:openai/gpt-oss-120b")

# Qual variável de ambiente cada provedor precisa. Só usamos isto para dar uma
# mensagem de erro amigável antes de o framework reclamar.
CHAVES = {"groq": "GROQ_API_KEY", "google-gla": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}

# Substitui o MAX_PASSOS do nível 1: no máximo 8 chamadas ao modelo por pergunta.
LIMITES = UsageLimits(request_limit=8)


def verificar_chave() -> None:
    """Para o programa com uma mensagem clara se a chave do provedor não estiver definida."""
    provedor = MODELO.split(":", 1)[0]
    variavel = CHAVES.get(provedor)
    if variavel and not os.environ.get(variavel):
        raise SystemExit(f"Defina a variável de ambiente {variavel}.")


verificar_chave()

# ---------------------------------------------------------------------------
# 2. Agente
# ---------------------------------------------------------------------------
# `instructions` é o equivalente ao PROMPT_SISTEMA do nível 1.
# `retries=2`: quando uma ferramenta levanta ModelRetry (ou recebe argumentos
# inválidos), o framework devolve o erro ao modelo e deixa ele tentar até 2 vezes.

agente = Agent(
    MODELO,
    instructions=(
        "Você é um assistente prestativo que responde em português. "
        "Use as ferramentas disponíveis quando precisar de contas, da data atual "
        "ou de informações factuais. Não invente resultados de ferramentas. "
        "Nunca faça contas de cabeça: use 'calcular' para números e 'dias_entre_datas' para datas. "
        "Se uma ferramenta falhar, diga ao usuário que não conseguiu; "
        "nunca responda com um valor que nenhuma ferramenta confirmou."
    ),
    retries=2,
)

# ---------------------------------------------------------------------------
# 3. Ferramentas
# ---------------------------------------------------------------------------
# @agente.tool_plain registra a função como ferramenta. O framework lê:
#   - o NOME da função        → nome da ferramenta;
#   - a DOCSTRING              → descrição que o modelo vê (diga QUANDO usar);
#   - a seção "Args:"          → descrição de cada argumento;
#   - os TYPE HINTS            → tipos dos argumentos, que o Pydantic valida.
# Ou seja: o ESQUEMA_FERRAMENTAS do nível 1 agora sai do próprio código.
#
# "plain" = a ferramenta não precisa de contexto da execução. Existe também
# @agente.tool, que recebe um RunContext (útil para passar conexão de banco,
# usuário logado etc.); veremos quando precisar.

_OPERADORES = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _avaliar(no):
    """Avalia um nó da AST aceitando só números e operadores aritméticos (ver nível 1)."""
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return no.value
    if isinstance(no, ast.BinOp) and type(no.op) in _OPERADORES:
        return _OPERADORES[type(no.op)](_avaliar(no.left), _avaliar(no.right))
    if isinstance(no, ast.UnaryOp) and type(no.op) in _OPERADORES:
        return _OPERADORES[type(no.op)](_avaliar(no.operand))
    raise ValueError("expressão não suportada")


@agente.tool_plain
def calcular(expressao: str) -> str:
    """Calcula uma expressão aritmética. Use sempre que precisar fazer contas.

    Args:
        expressao: Expressão só com números e operadores, ex.: (12.5 * 4) / 3
    """
    try:
        return str(_avaliar(ast.parse(expressao, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        # ModelRetry manda esta mensagem de volta ao modelo como "tente de novo".
        # No nível 1 nós devolvíamos "ERRO ..." como texto e torcíamos para ele entender.
        raise ModelRetry(
            f"Expressão inválida ({e}). Use apenas números e os operadores + - * / ** % //."
        ) from e


DIAS_SEMANA = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
               "sexta-feira", "sábado", "domingo"]


@agente.tool_plain
def data_hora_atual() -> str:
    """Retorna a data e a hora atuais no formato AAAA-MM-DD HH:MM e o dia da semana."""
    agora = datetime.now()
    return f"{agora:%Y-%m-%d %H:%M} (formato ano-mês-dia), {DIAS_SEMANA[agora.weekday()]}"


@agente.tool_plain
def dias_entre_datas(data_inicial: date, data_final: date) -> str:
    """Calcula quantos dias há entre duas datas.

    Use para qualquer pergunta do tipo 'quantos dias faltam' ou 'quantos dias se passaram'.

    Args:
        data_inicial: Data de início, formato AAAA-MM-DD.
        data_final: Data de fim, formato AAAA-MM-DD.
    """
    # Repare: não há nenhuma conversão de texto para data aqui. Como os argumentos
    # têm o type hint `date`, o Pydantic converte "2026-11-10" em date(2026, 11, 10)
    # antes de chamar a função. Se o modelo mandar "10/11/2026", a validação falha
    # e o framework pede ao modelo que corrija, sem esta função nem ser chamada.
    return f"{(data_final - data_inicial).days} dias de {data_inicial} até {data_final}"


@agente.tool_plain
def buscar_wikipedia(termo: str) -> str:
    """Busca um assunto na Wikipédia em português e retorna um resumo.

    Args:
        termo: Assunto a pesquisar.
    """
    cabecalho = {"User-Agent": "agente-nivel-2/1.0 (projeto de estudo)"}
    try:
        busca = "https://pt.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "opensearch", "search": termo, "limit": 1, "format": "json"}
        )
        with urllib.request.urlopen(urllib.request.Request(busca, headers=cabecalho), timeout=15) as r:
            titulos = json.load(r)[1]
        if not titulos:
            return "Nenhum artigo encontrado."

        resumo = "https://pt.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(titulos[0])
        with urllib.request.urlopen(urllib.request.Request(resumo, headers=cabecalho), timeout=15) as r:
            dados = json.load(r)
        return f"{dados.get('title')}: {dados.get('extract', '(sem resumo)')}"
    except OSError as e:
        # Falha de rede não se resolve tentando de novo com outro termo,
        # então devolvemos texto (sem ModelRetry) para o modelo avisar o usuário.
        return f"Não foi possível acessar a Wikipédia: {e}"


# ---------------------------------------------------------------------------
# 4. Interface de terminal
# ---------------------------------------------------------------------------

def mostrar_ferramentas(mensagens) -> None:
    """Imprime as ferramentas usadas numa execução, como o 🔧 do nível 1.

    O framework guarda o histórico como objetos (não dicts). Cada mensagem tem
    "partes": pedidos de ferramenta (ToolCallPart), resultados (ToolReturnPart),
    pedidos de correção (RetryPromptPart), texto etc.
    """
    for mensagem in mensagens:
        for parte in mensagem.parts:
            if isinstance(parte, ToolCallPart):
                print(f"  🔧 {parte.tool_name}({parte.args_as_json_str()})")
            elif isinstance(parte, ToolReturnPart):
                print(f"     → {str(parte.content)[:120]}")
            elif isinstance(parte, RetryPromptPart) and parte.tool_name:
                print(f"     ↩ {parte.tool_name} falhou, o modelo vai tentar de novo")


def main():
    """Chat no terminal.

    O histórico agora é uma lista de mensagens do próprio framework.
    `resultado.all_messages()` devolve a conversa inteira até aqui, que passamos
    de volta na próxima pergunta. Se run_sync falhar, `historico` simplesmente
    não é atualizado: não precisamos mais do rollback manual do nível 1.
    """
    print(f"Agente pronto ({MODELO}). Digite 'sair' para encerrar.\n")
    historico = []
    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        try:
            resultado = agente.run_sync(pergunta, message_history=historico, usage_limits=LIMITES)
        except UsageLimitExceeded:
            print("⚠️  Parei: atingi o limite de chamadas ao modelo sem chegar a uma resposta.\n")
            continue
        except Exception as e:  # erros de API (401, 429...), rede, retries esgotados
            print(f"⚠️  {type(e).__name__}: {e}\n")
            continue

        mostrar_ferramentas(resultado.new_messages())  # só as mensagens desta pergunta
        historico = resultado.all_messages()
        print(f"Agente: {resultado.output}\n")


if __name__ == "__main__":
    main()