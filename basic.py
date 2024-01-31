# #####SYMBOLS#####
SYM_PLUS = '+'
SYM_MINUS = '-'
SYM_MUL = '*'
SYM_DIV = '/'
SYM_POWER = '^'
SYM_REMAINDER = '%'
SYM_EQUAL = '='
SYM_KEYWORD = 'KEYWORD'
SYM_IDENTIFIER = 'IDENTIFIER'
SYM_INT = 'INT'
SYM_FLOAT = 'FLOAT'
SYM_RPARENTHESIS = '('
SYM_LPARENTHESIS = ')'
SYM_EOF = 'EOF'
# #####CONSTANTS#####
DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
KEYWORDS = ['LET', 'IF', 'THEN', 'ELSE', 'REPEAT', 'UNTIL', 'GOSUB', 'SUB', 'RETURN', 'PRINT']


# #####CLASSES#####
###################################################
# ERROR CLASS
###################################################
class Error:
    def __init__(self, error_type, detail, pos_start, pos_end):
        self.error_type = error_type
        self.detail = detail
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __str__(self):
        return f'{self.error_type}: {self.detail} File {self.pos_start.fn}, line {self.pos_start.ln + 1}'


class IllegalCharacterError(Error):
    def __init__(self, detail, pos_start, pos_end):
        super().__init__('Illegal Character Error', detail, pos_start, pos_end)


class InvalidSyntaxError(Error):
    def __init__(self, detail, pos_start, pos_end):
        super().__init__('Invalid Syntax', detail, pos_start, pos_end)


class RTError(Error):
    def __init__(self, detail, pos_start, pos_end, context):
        super().__init__('Run Time Error', detail, pos_start, pos_end)
        self.context = context

    def __str__(self):
        result = self.generate_traceback()
        result += f'{self.error_type}: {self.detail}\n'
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context
        while ctx is not None:
            result += f' File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n'
            pos = self.context.parent_entry_pos
            ctx = self.context.parent
        return 'Traceback (most recent call last):\n' + result


###################################################
# SYMBOL CLASS
###################################################
class Symbol:
    def __init__(self, types, value=None, pos_start=None, pos_end=None):
        self.types = types
        self.value = value
        if pos_start is not None:
            self.pos_start = pos_start
            self.pos_end = pos_start.advance()
        if pos_end is not None:
            self.pos_end = pos_end.copy()

    def matches(self, value, types):
        return self.value == value and self.types == types

    def __repr__(self):
        if self.value is not None:
            return f'{self.types}: {self.value}'
        else:
            return f'{self.types}'


###################################################
# SCANNER
###################################################
class Scanner:
    def __init__(self, text: str, fn):
        self.fn = fn
        self.text = text
        self.pointer = Position(-1, 0, -1, fn, text)
        self.current_character = None
        self.advance()

    def advance(self):
        self.pointer.advance(self.current_character)
        if self.pointer.idx < len(self.text):
            self.current_character = self.text[self.pointer.idx]
        else:
            self.current_character = None

    def make_token(self):
        symbols = []
        while self.current_character is not None:
            if self.current_character in ' \t':
                self.advance()
            elif self.current_character == '+':
                symbols.append(Symbol(SYM_PLUS, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '-':
                symbols.append(Symbol(SYM_MINUS, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '*':
                symbols.append(Symbol(SYM_MUL, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '/':
                symbols.append(Symbol(SYM_DIV, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '(':
                symbols.append(Symbol(SYM_LPARENTHESIS, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == ')':
                symbols.append(Symbol(SYM_RPARENTHESIS, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '^':
                symbols.append(Symbol(SYM_POWER, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '%':
                symbols.append(Symbol(SYM_REMAINDER, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character == '=':
                symbols.append(Symbol(SYM_EQUAL, pos_start=self.pointer.copy()))
                self.advance()
            elif self.current_character in DIGITS:
                types, value, error, pos_start, pos_end = self.make_number()
                if error is None:
                    symbols.append(Symbol(types, value, pos_start, pos_end))
                else:
                    return [], error
            elif self.current_character in LETTERS:
                types, value, error, pos_start, pos_end = self.make_identifier()
                symbols.append(Symbol(types, value, pos_start, pos_end))
            else:
                current_char = self.current_character
                pos_start = self.pointer.copy()
                pos_end = self.pointer.advance(self.current_character).copy()
                return [], IllegalCharacterError("'" + current_char + "'", pos_start, pos_end)
        symbols.append(Symbol(SYM_EOF, pos_start=self.pointer.copy()))
        return symbols, None

    def make_number(self):
        pos_start = self.pointer.copy()
        num_str = ''
        dot_count = 0
        while self.current_character is not None and self.current_character in (DIGITS + '.'):
            if dot_count > 1:
                pos_start = self.pointer.copy()
                pos_end = self.pointer.advance(self.current_character).copy()
                return None, None, IllegalCharacterError('A number cannot have 2 dots.', pos_start, pos_end)
            if self.current_character == '.':
                dot_count += 1
            num_str += self.current_character
            self.advance()
        if dot_count == 0:
            return SYM_INT, int(num_str), None, pos_start, self.pointer
        else:
            return SYM_FLOAT, float(num_str), None, pos_start, self.pointer

    def make_identifier(self):
        pos_start = self.pointer.copy()
        id_str = ''
        while self.current_character is not None and self.current_character in (DIGITS + LETTERS + '_'):
            id_str += self.current_character
            self.advance()
        if id_str in KEYWORDS:
            return SYM_KEYWORD, id_str, None, pos_start, self.pointer.copy()
        else:
            return SYM_IDENTIFIER, id_str, None, pos_start, self.pointer.copy()


###################################################
# POSITION
###################################################
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1
        if current_char == '\n':
            self.ln += 1
            self.col = 0
        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


###################################################
# NODES
###################################################
class NumberNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class UnOpNode:
    def __init__(self, op_tok, right_node):
        self.op_tok = op_tok
        self.right_node = right_node
        self.pos_start = self.op_tok.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.right_node})'


    

###################################################
# PARSER
###################################################
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.current_tok = None
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    ###################################################
    def parse(self):
        res = self.arithmetic()
        if res.error is not None and self.current_tok.types != SYM_EOF:
            return res.failure(InvalidSyntaxError(
                pos_start=self.current_tok.pos_start, pos_end=self.current_tok.pos_end,
                detail="Expected '+', '-', '*' or '/'"
            ))
        return res

    def variable(self):
        res = ParseResult()
        if self.current_tok.matches(SYM_KEYWORD, 'LET'):
            res.register(self.advance())
            if self.current_tok.types != SYM_IDENTIFIER:
                return res.failure(InvalidSyntaxError(f'Expected Identifier but found {self.current_tok.types}.',
                                                      self.current_tok.pos_start.copy(),
                                                      self.current_tok.pos_end.copy()))
            else:
                var_name = self.current_tok
                res.register(self.advance())
                if self.current_tok.types != SYM_EQUAL:
                    return res.failure(InvalidSyntaxError(f"Expected '=' but found {self.current_tok.types}.",
                                                          self.current_tok.pos_start.copy(),
                                                          self.current_tok.pos_end.copy()))
                else:
                    res.register(self.advance())
                    right = res.register(self.arithmetic())
                    if res.error is not None:
                        return res
                    else:
                        return res.success(right)
        else:
            right = res.register(self.arithmetic())
            if res.error is not None:
                return res
            else:
                return res.success(right)

    def arithmetic(self):
        res = ParseResult()
        left = res.register(self.term())
        if res.error is not None:
            return res
        while self.current_tok.types in (SYM_PLUS, SYM_MINUS):
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(self.term())
            if res.error is not None:
                return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)

    def term(self):
        res = ParseResult()
        left = res.register(self.factor())
        if res.error is not None:
            return res
        while self.current_tok.types in (SYM_MUL, SYM_DIV, SYM_POWER, SYM_REMAINDER):
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(self.factor())
            if res.error is not None:
                return res
            left = BinOpNode(left, op_tok, right)
        return res.success(left)

    def factor(self):
        res = ParseResult()
        tok = self.current_tok
        if tok.types in (SYM_INT, SYM_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))
        elif tok.types in (SYM_PLUS, SYM_MINUS):
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(self.arithmetic())
            right = UnOpNode(op_tok, right)
            return res.success(right)
        elif tok.types == SYM_LPARENTHESIS:
            res.register(self.advance())
            arithmetic = res.register(self.arithmetic())
            if res.error is not None:
                return res
            if self.current_tok.types == SYM_RPARENTHESIS:
                res.register(self.advance())
                return res.success(arithmetic)
            else:
                return res.failure(
                    InvalidSyntaxError(pos_start=self.current_tok.pos_start, pos_end=self.current_tok.pos_end,
                                       detail="Expected ')'"))
        return res.failure(InvalidSyntaxError("Expected int or float or an expression", tok.pos_start, tok.pos_end))
    ###################################################


###################################################
# PARSE RESULT
###################################################
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error is not None:
                self.error = res.error
            return res.node
        return res

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


###################################################
# VALUES
###################################################
class Number:
    def __init__(self, value):
        self.pos_start = None
        self.pos_end = None
        self.context = None
        self.value = value

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_to(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_to(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_to(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError('Division by 0 Error.', other.pos_start, other.pos_end, self.context)
            return Number(self.value / other.value).set_context(self.context), None

    def pow_to(self, other):
        if isinstance(other, Number):
            if self.value == 0 and other.value == 0:
                return None, RTError('Zero to the power of Zero Error', self.pos_start, other.pos_end, self.context)
            return Number(self.value ** other.value).set_context(self.context), None

    def rem_to(self, other):
        if isinstance(other, Number):
            return Number(self.value % other.value).set_context(self.context), None

    def __repr__(self):
        return f'{self.value}'


###################################################
# RUNTIME RESULT
###################################################
class RTResult:
    def __init__(self):
        self.error = None
        self.value = None

    def register(self, res):
        if isinstance(res, RTResult):
            if res.error is not None:
                self.error = res.error
            return res.value
        return res

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


###################################################
# CONTEXT
###################################################
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos


###################################################
# INTERPRETER
###################################################
class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node):
        raise Exception(f'No visit{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        error = None
        result = None
        left = res.register(self.visit(node.left_node, context))
        if res.error is not None:
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.error is not None:
            return res

        if node.op_tok.types == SYM_PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.types == SYM_MINUS:
            result, error = left.subbed_to(right)
        elif node.op_tok.types == SYM_MUL:
            result, error = left.multed_to(right)
        elif node.op_tok.types == SYM_DIV:
            result, error = left.dived_to(right)
        elif node.op_tok.types == SYM_POWER:
            result, error = left.pow_to(right)
        elif node.op_tok.types == SYM_REMAINDER:
            result, error = left.rem_to(right)
        if error is not None:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnOpNode(self, node, context):
        res = RTResult()
        error = None
        number = res.register(self.visit(node.right_node, context))
        if node.op_tok.types == SYM_MINUS:
            number, error = number.multed_to(Number(-1)), None
        if error is not None:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


# #####PROCEDURES#####
###################################################
# RUN
###################################################
def run(text, fn):
    # GENERATE SYMBOLS
    scanner = Scanner(text, fn)
    symbols, error = scanner.make_token()
    if error is not None:
        return None, error
    # GENERATE SYNTAX TREE
    parser = Parser(symbols)
    ast = parser.parse()
    if ast.error is not None:
        return None, ast.error
    else:
        return ast.node, None
    # CREATE AN INTERPRETER INSTANCE
    #interpreter = Interpreter()
    #context = Context('<program>')
    #visit = interpreter.visit(ast.node, context)
    #return visit.value, visit.error
