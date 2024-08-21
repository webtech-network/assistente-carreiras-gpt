## Fluxo de Trabalho

### Metodologia
- Semanalmente gerar focos de estudo e testes.
- Testar e iterar continuamente.

### Problemas 
- O assistente confunde os nomes dos cursos e atributos dos cursos.

### Estratégias
- FineTuning.
    - Acostumar mais o modelo ao tipo de textos e informações desejadas.
    - Ajuste de saídas desejadas vs indesejadas.
- Rag.
    - Melhoria no retrieval de documentos.
    - Melhorias na arquitetura e hiperparâmetros.

### Divisão de Responsabilidades

#### Cláudio Augusto
- Estudar e entender embeddings para orientação do resto do grupo.
    - Deeplearning.ai (https://learn.deeplearning.ai/courses/embedding-models-from-architecture-to-implementation/lesson/3/contextualized-token-embeddings).
- Estudar e entender sistemas e arquitetura rag para orientação do resto do grupo e programação.
    - Deeplearning.ai.
    - ML Street Talk.
- Estudar langsmith para processo de testes e tracking (reunião na próxima semana).
- GridSearch (otimização de parâmetros para a saída).
- Ajudar na documentação.
- Organizar Kanban.

#### Henrique Nahim
- Estudar langsmith para processo de testes e tracking (reunião na próxima semana).
- WorkShop de assistente (lab).
- Checar alterações no formato do texto de entrada para melhorar a saída.
	- Ver se é melhor 1 pdf ou os 30.
	- Ver se o LangChain ou a OpenAI (por baixo dos panos) oferece alguma ferramenta para dar mais importância a certas partes do texto (mecanismos de atenção.)
		- **Objetivo:** corrigir o bug em que o modelo oferece na saída, algum atributo no lugar do nome do curso ou vice versa (nome do curso no lugar de um atributo).

#### Gabriel
- Estudar langsmith para processo de testes e tracking (reunião na próxima semana).
- Estudar iterações de prompt.