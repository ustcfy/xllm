/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "api_service/rerank_service_impl.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

class RerankServiceImplTestPeer final : public RerankServiceImpl {
 public:
  using RerankServiceImpl::has_documents;
};

TEST(RerankServiceImplTest, HasDocumentsRejectsEmptyRequest) {
  proto::RerankRequest request;

  EXPECT_FALSE(RerankServiceImplTestPeer::has_documents(request));

  request.add_documents("document");
  EXPECT_TRUE(RerankServiceImplTestPeer::has_documents(request));
}

}  // namespace
}  // namespace xllm
