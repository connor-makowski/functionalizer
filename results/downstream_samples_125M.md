# Downstream Generation Samples Report (125M256)

This report showcases a comparison of generated outputs between the different tokenizer configurations.

For code datasets, samples are labelled by whether the generation passed the syntax check; a collapsed generation is a failure regardless of what the checker would have said, so each sample also reports whether it collapsed. Prose generation has no pass/fail criterion, so two representative generations are shown instead.

## Dataset: FineWeb-Edu (Natural Language Generation)

### Configuration: Llama Split Only

#### Example Generation 1
* **Prompt**: `'A recent report from the Worldwatch Institute'`
* **Generated Tokens (first 20)**: `[',', ' which', ' found', ' that', ' the', ' number', ' of', ' people', ' who', ' have', ' been', ' diagnosed', ' with', ' HIV', ' has', ' increased', ' by', ' ', '1', '0', '...']`
* **Decoded Completion**: `', which found that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000.\nThe report also found that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000.\nThe report also found that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000.\nThe report also found that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000.\nThe report also found that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000.\nThe report also found that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that the number of people who have been diagnosed with HIV has increased by 10 percent since 2000, and that number has increased by 10 percent since 2000, and that number has increased by 10 percent since 2000, and that number has increased by 10 percent since 2000, and that number'`
* **Completion Length (chars)**: 2240
* **Collapsed**: yes (tokens pre-collapse: 514)

#### Example Generation 2
* **Prompt**: `'April 12, 2004.\nBy Salynn Boyles,\nMedical News'`
* **Generated Tokens (first 20)**: `[' Service', ',', ' ', '2', '0', '0', '4', '\n', 'The', ' American', ' Medical', ' Association', ' (', 'A', 'HA', ')', ' is', ' a', ' nonpro', 'fit', '...']`
* **Decoded Completion**: `' Service, 2004\nThe American Medical Association (AHA) is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients and their families. AHA is a nonprofit organization that promotes the use of medical information to improve the quality of life for patients'`
* **Completion Length (chars)**: 5528
* **Collapsed**: no (tokens pre-collapse: 1024)

---

### Configuration: Llama Split + Functionalizer

#### Example Generation 1
* **Prompt**: `'A recent report from the Worldwatch Institute'`
* **Generated Tokens (first 20)**: `[',', '\ue100\ue001', '\ue100\ue002', '\ue100\ue003', '\ue100\ue004', '\ue100\ue005', '\ue100\ue006', ' un', 'icef', ',', ' and', ' the', '\ue100\ue001', ' world', 'watch', '\ue100\ue001', ' institute', ',', '\ue100\ue001', '\ue100\ue002', '...']`
* **Decoded Completion**: `', UNICEF, and the Worldwatch Institute, UNICEF, found that the number of children in the UK is growing at a rate of 1.5 per cent. The UK’s UNICEF report, UNICEF, found that 1.5 per cent of children in the UK are living in poverty, and 1.5 per cent are living in poverty.\nThe report, UNICEF, found that 1.5 per cent of children in the UK are living in poverty, and 1.5 per cent are living in poverty. The report also found that 1.5 per cent of children in the UK are living in poverty, and 1.5 per cent are living in poverty.\nThe report also found that 1.5 per cent of children in the UK are living in poverty, and 1.5 per cent are living in poverty, and 1.5 per cent are living in poverty, and 1.5 per cent are living in poverty, and 1.5 per cent'`
* **Completion Length (chars)**: 746
* **Collapsed**: yes (tokens pre-collapse: 220)

#### Example Generation 2
* **Prompt**: `'April 12, 2004.\nBy Salynn Boyles,\nMedical News'`
* **Generated Tokens (first 20)**: `['\ue100\ue001', ' service', '\n', '\ue100\ue000', 'the', '\ue100\ue001', '\ue100\ue002', '\ue100\ue003', '\ue100\ue004', '\ue100\ue005', '\ue100\ue006', '\ue100\ue007', '\ue100\ue008', '\ue100\ue009', '\ue100\ue00a', '\ue100\ue00b', '\ue100\ue00c', '\ue100\ue00d', '\ue100\ue00e', '\ue100', '...']`
* **Decoded Completion**: `' Service\nThe'`
* **Completion Length (chars)**: 12
* **Collapsed**: yes (tokens pre-collapse: 31)

---

## Dataset: github-code-python (Code Generation - Python)

### Configuration: Llama Split Only

#### Success Sample
* **Prompt**: `'class PrivateKeyError(OpenSSLObjectError):'`
* **Generated Tokens (first 20)**: `[' ', ' #', ' p', 'ra', 'g', 'm', 'a', ':', ' ', 'N', 'O', ' ', 'C', 'O', 'V', 'E', 'R', '\n', '   ', ' "', '...']`
* **Decoded Completion**: `'  # pragma: NO COVER\n    """\n    Raised when an error occurs.\n    """\n    raise NotImplementedError("PrivateKeyError")\n\nclass PrivateKeyError(OpenSSLObjectError):\n    """\n    Raised when an error occurs.\n    """\n    raise NotImplementedError("PrivateKeyError")\n\nclass PrivateKeyError(OpenSSLObjectError):\n    """\n    Raised when an error occurs.\n    """\n    raise NotI'`
* **Completion Length (chars)**: 368
* **Collapsed**: no (tokens pre-collapse: 256)

#### Failure Sample
* **Prompt**: `'class Bgp_global(ResourceModule):'`
* **Generated Tokens (first 20)**: `[' ', ' #', ' ', 'S', 't', 'r', 'ing', '\n', '\t', '\t', 'self', '.', 'ad', 'd', '_', 'qu', 'er', 'y', '_p', 'ar', '...']`
* **Decoded Completion**: `"  # String\n\t\tself.add_query_param('ResourceModule', ResourceModule)\n\tdef get_ResourceOwnerId(self): # Long\n\t\treturn self.get_query_params().get('ResourceOwnerId')\n\n\tdef set_ResourceOwnerId(self, ResourceOwnerId):  # Long\n\t\tself.add_query_param('ResourceOwnerId', ResourceOwnerId)\n\tdef get_ResourceOwnerAccount(self): # String\n\t\treturn self.get_query_params().get('ResourceOwnerAccount')\n\n"`
* **Completion Length (chars)**: 388
* **Collapsed**: no (tokens pre-collapse: 256)

---

### Configuration: Llama Split + Functionalizer

#### Success Sample
* **Prompt**: `'def make_file_unblocking(file):'`
* **Generated Tokens (first 20)**: `[' ', ' ', '#', ' p', 'y', 'l', 'in', 't', ':', ' ', 'd', 'is', 'a', 'b', 'le', '=', 'un', 'u', 'se', 'd', '...']`
* **Decoded Completion**: `'  # pylint: disable=unused-argument\n    """Make a file unblocking file."""\n    file_unblocking = file.read()\n    if file_unblocking:\n        return file_unblocking\n    else:\n        return file_unblocking\n\n\ndef make_file_unblocking(file):  # pylint: disable=unused-argument\n    """Make a file unblocking file."""\n    file_unblocking = file.read()\n    if file_unblocking:\n        return file_unblocking\n    '`
* **Completion Length (chars)**: 406
* **Collapsed**: no (tokens pre-collapse: 256)

#### Failure Sample
* **Prompt**: `'class Bgp_global(ResourceModule):'`
* **Generated Tokens (first 20)**: `[' ', ' ', '#', '\ue100\ue001', ' ', 're', 's', 'o', 'ur', 'ce', ' ', 'mo', 'd', 'u', 'le', '\n', '\ue200\ue000\ue004', ' ', '\ue200\ue000\ue003', '"', '...']`
* **Decoded Completion**: `'  # Resource module\n    """Bgp global Bgp global Bgp global Bgp global B'`
* **Completion Length (chars)**: 72
* **Collapsed**: yes (tokens pre-collapse: 31)

---

