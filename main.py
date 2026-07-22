from models import Parser

if __name__ == "__main__":
    func_list = Parser.parse("src/data/input/functions_definition.json")
    print(func_list)
