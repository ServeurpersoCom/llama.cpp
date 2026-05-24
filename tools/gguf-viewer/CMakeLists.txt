set(TARGET llama-gguf-viewer)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(TARGET_SRCS
    viewer.cpp
)

set(PUBLIC_ASSETS
    app.js
    architecture.js
    browser.js
    common.js
    heatmap.js
    index.html
    metadata.js
    statistics.js
    tensors.js
    tokenizer.js
    viewer.css
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${CMAKE_CURRENT_SOURCE_DIR}/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

# ensure common headers available
include_directories(${CMAKE_SOURCE_DIR})

find_package(Threads REQUIRED)

target_link_libraries(${TARGET} PRIVATE llama-common llama cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
