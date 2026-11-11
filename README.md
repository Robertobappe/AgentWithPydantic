# Agente nível 2 — Pydantic AI

O mesmo agente do [nível 1](../nivel1/), com as mesmas 4 ferramentas, reescrito com o
framework [Pydantic AI](https://pydantic.dev/docs/ai/). O objetivo é ver, lado a lado,
o que um framework leve assume no lugar do código escrito à mão.

## Como rodar

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."
python agente.py
```

Trocar de modelo: `MODELO="provedor:modelo" python agente.py`
(ex.: `groq:llama-3.3-70b-versatile`).

## O que o framework assumiu

| Responsabilidade | Nível 1 (na mão) | Nível 2 (Pydantic AI) |
|---|---|---|
| Chamar a API | `chamar_modelo()` com urllib | o framework chama |
| Descrever as ferramentas | `ESQUEMA_FERRAMENTAS` em JSON Schema | gerado dos type hints e da docstring |
| Registrar as ferramentas | dicionário `FERRAMENTAS` | `@agente.tool_plain` |
| Loop e limite de passos | `for` + `MAX_PASSOS` | `run_sync()` + `UsageLimits(request_limit=8)` |
| Erro de ferramenta | devolve `"ERRO ..."` como texto | `ModelRetry`: o modelo recebe o erro e tenta de novo |
| Validar argumentos | `date.fromisoformat()` dentro da função | type hint `date`; o Pydantic valida antes de chamar |
| Histórico | lista de dicts e rollback manual em falha | `message_history`; em falha o histórico não muda |

O código ficou com as **regras do agente** (instruções e ferramentas). O
**mecanismo** (HTTP, loop, schema, retries) passou para o framework.

## O que ganhamos e o que perdemos

**Ganhos**
- Menos código de infraestrutura e menos lugar para bug.
- Validação automática de argumentos: o modelo não consegue chamar
  `dias_entre_datas` com `"10/11/2026"`, o bug da v0.3 do nível 1.
- Retry estruturado: o modelo recebe o motivo da falha e corrige.
- Trocar de provedor é trocar uma string.

**Perdas**
- Uma dependência externa (`pydantic-ai-slim[groq]`).
- O mecanismo fica escondido. Sem o nível 1, seria difícil entender o que
  `run_sync()` faz por baixo.
- As ferramentas usadas só aparecem no terminal **depois** da resposta
  (lidas do histórico), não em tempo real como no nível 1.

## Próximos passos possíveis

- **Saída estruturada** (`output_type=`): o agente devolve um objeto Pydantic
  validado em vez de texto livre.
- **Testes sem gastar API** com `TestModel` / `FunctionModel` do próprio framework.
- **Ferramentas com contexto** (`@agente.tool` + `RunContext`) para injetar
  dependências como conexão de banco ou usuário logado.