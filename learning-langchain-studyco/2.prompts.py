from langchain.prompts import HumanMessagePromptTemplate

template = """\
次のコマンドの概要を説明してください。

コマンド: {command}
"""

prompt = HumanMessagePromptTemplate.from_template(template)
print(prompt.format(command="echo").content)
