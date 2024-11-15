THE 2BASIC LANGUAGE PROJECT: IBHI Ahmed Yassine

This my first project, it consists of writing a simplified version of BASIC based on python.
This project was inspired from the build-your-own-x repository.

This is the grammar of the 2BASIC (TOO BASIC) programming language.

2BASIC is a subset of the unstructured BASIC language. It features:

### File Extension:
I chose <file_name>.b as the convention to name a 2BASIC file.

### Data types and variables:
* 2BASIC supports integers, characters following the ASCII encoding, and pointers.
* variable names can be any letters long.
* Arrays are supported but other data types (linked lists, dictionaries ...) aren't supported natively in 2BASIC.
* comments declared with `//` are single-line always and are ignored in the language.

### Keywords:
`IF`, `THEN`, `ELSE`, `REPEAT`, `UNTIL`, `WITH`, `GOSUB`, `SUB`, `RETURN`, `PRINT`, `TRUE`, `FALSE`

### Symbols:
`integer`, `string`, `identifier`, `(`, `)`, `+`, `-`, `*`, `/`, `%`, `^`, `=`, `==`, `!=`, `<`, `>`, `<=`, `>=` , `AND`, `OR`, `NOT`

with:

````
integer     = digit { digit }.
    
string      = "'" printable_character "'" | """ printable_character """.

identifier  = letter { letter | digit | "_" }.
````

and:

```` 
digit  = "0" | ... | "9" .

letter = "a" | ... | "z" | "A" | ... | "Z" .

compaaison_op =  "==" | "!=" | "<" | ">" | "<=" | ">=" .
````

### Grammar:

````
basic       = { variable | block }.

variable = "LET" identifier "=" (logical_expression).

logical_expression = ["NOT"] ( (comp_expression) [ ("AND" | "OR" ) ( logical_expression) ] ).
    
<<<<<<< HEAD
comp_expression  = (arithmetic) [ comparaison_op (arithmetic) ].
=======
expression  = arithmetic [ ( "==" | "!=" | "<" | ">" | "<=" | ">=" | "AND" | "OR" | "NOT" ) arithmetic ].
>>>>>>> master

arithmetic  = term { ( "+" |" -" ) term }.

term        = factor { ( "*" | "/" | "%" | "^" ) factor }.
 
factor      = [ "+" | "-" ] ( integer | identifier | "(" arithmetic ")" | if ) .
 
literal     = integer | string .
 
statement   = assignment | if | loop | call | return .

assignment  = identifier "=" literal. 
    
if          = "IF" logical_expression "THEN" "(" arithmetic | statment ")" 
               {"ELIF" logical_expression "THEN" "(" arithmetic | statment ")" } 
               ["ELSE" "(" arithmetic | statement ")" ].
    
loop        = "REPEAT" statement | arithmetic
              "UNTIL" logical_expression
              "WITH" arithmetic.
    
block       = "SUB" variable ":" [variable [ { "," variable } ] ]
               statement
               "RETURN" [expression].
              
call        = [variable] "GOSUB" identifier [":" variable].

log         = "PRINT" ( expression ).
````


