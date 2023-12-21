import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=0
    )
    ls = [
        """到、实现数据权限和功能权限配置。
支持对用户进行新增、修改、删除及用户角色分配和数据权限设置，并可对账号状态进行停用、启用管理，停用后改账号将无法登录平台。
"	AIOT平台用户管理
组织管理主要是通过组织树的方式对平台用户进行组织机构配置管理，支持用户组织树查询、新增、修改、删除操作，方便用户根据本单位的用户组织结构灵活设置。	AIOT平台组织管理
用户开户后系统会提供一个默认密码，用户可根据自身需要定期修改密码，系统支持修改密码时输入原密码验证，并对密码字符及长度要求合法性校验，以提高用户账号密码的安全性。	AIOT平台密码修改
平台提供全面开放的接口服务，系统集成公司可根据API授权获取对应的接口授权信息，实现接口对接。可通过创建应用的方式对接多个系统，创建应用时可获取接口秘钥，获取秘钥后参考接口文档进行对接。	AIOT平台API授权
主要针对系统级参数进行设置，参数设置一般都有默认参数，修改的频率较低，且有些参数设置需要具有一定的技术基础的人员配置，主要用以对客户的个性化参数进行配置，提高系统的灵活性。包括：二维码注册有效期、微信公众号ID，回调接口地址等。	AIOT平台系统参数
展示门禁终端的屏保管理服务，包括屏保图片和设备关联管理服务，支持停用/启用屏保，支持删除屏保服务。	AIOT平台屏保管理
"传统ISV智能化升级改造，利用AI综合能力平台赋能，或基于平台重新构建能力平台，如：通过AI能力平台实现各种行为分析，轨迹刻画，布控，智能检索等；其它细分行业软件公司：通过小视智能化平台获取的物联传感、人脸、视频结构化等数据进行场景化应用，如：办公考勤、化工安全、工业质检、商圈零售、智慧教育等。其主要特点如下：
强能力，轻应用：基于平台的强能力、轻应用特性可以很好的适配更多的应用场景，并可基于该特性开发出更为细分场景化业务应用；
全能力输出，易对接：标准化的南北向接口输出，可快速实现对接，减少客户的学习成本；
高稳定性：通过微服务架构，模块间独立，事故的影响率较少，减少维护成本；
易扩展：通过分层和微组组合的方式，针对不同的能力诉求可快速扩展相应的能力；
私有化部署成本低：根据客户的诉求可灵活配置相应的能力，最大程度降低部署成本；
小场景业务闭环：基于AI能力实现单场景的业务闭环。
"	AIOT平台产品特点
"1.《计算机软件著作权登记证书》
2.《信息系统安全登记保护》第三级
3.《信创适配认证》
"	AIOT平台产品资质""",
        ]
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
