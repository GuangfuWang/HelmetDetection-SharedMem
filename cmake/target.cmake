add_library(${DEPLOY_LIB_NAME} SHARED ${LIB_SRC})
target_include_directories(${DEPLOY_LIB_NAME} PUBLIC ${CUDA_INCLUDE_DIR})
target_link_libraries(${DEPLOY_LIB_NAME} PUBLIC ${DEP_LIBS})

add_executable(${DEPLOY_MAIN_NAME} ${LIB_HEADER} ${LIB_MAIN})
target_link_libraries(${DEPLOY_MAIN_NAME} PUBLIC ${DEP_LIBS} ${DEPLOY_LIB_NAME})

add_executable(${DEPLOY_MPS} ${LIB_HEADER} ${EXE_MPS})
target_link_libraries(${DEPLOY_MPS} PUBLIC ${DEP_LIBS} ${DEPLOY_LIB_NAME})

add_executable(${DEPLOY_MAIN_NAME}_test ${LIB_HEADER} ${LIB_MAIN_TEST})
target_link_libraries(${DEPLOY_MAIN_NAME}_test PUBLIC ${DEP_LIBS} ${DEPLOY_LIB_NAME})


