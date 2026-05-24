# embed a file as a C byte array header, the way xxd produces one
# invoked per asset with INPUT and OUTPUT defined on the cmake command line

get_filename_component(filename "${INPUT}" NAME)
string(MAKE_C_IDENTIFIER "${filename}" name)

file(READ "${INPUT}" hex_data HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_sequence "${hex_data}")

string(LENGTH "${hex_data}" hex_len)
math(EXPR len "${hex_len} / 2")

file(WRITE "${OUTPUT}" "unsigned char ${name}[] = {${hex_sequence}};\nunsigned int ${name}_len = ${len};\n")
