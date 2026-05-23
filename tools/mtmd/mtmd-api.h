// Export macro for the mtmd shared library. Kept in its own tiny header with no
// dependencies so the self contained modules (code2wav, talker) can export their
// symbols across the windows dll boundary without pulling in the full mtmd api.

#pragma once

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_API __declspec(dllexport)
#        else
#            define MTMD_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_API
#endif
