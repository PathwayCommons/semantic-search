import pytest
import types
from fastapi.exceptions import HTTPException

from semantic_search.ncbi import _medline_to_docs, _safe_request, _parse_medline, _get_eutil_records, uids_to_docs, Settings

settings = Settings()

def test_invalid_uid_test():
    with pytest.raises(HTTPException):
        uid = ["93846392868"]
        records = [{"id:": [uid]}]
        _medline_to_docs(records)

def test_safe_request():
    eutils_params = {
        "db": "pubmed",
        "id": "9887103",
        "retstart": 0,
        "retmode": "xml",
        "api_key": settings.ncbi_eutils_api_key,
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    assert _safe_request(url, "POST", files = eutils_params).status_code == 200

def test_get_eutil_records():
    eutil = "efetch"
    id = "9887103"
    #checking if generator is returned, need to check integrity of the returned value
    assert isinstance(_get_eutil_records(eutil, id), types.GeneratorType)

def test_parse_medline():
    text = "\nPMID- 9887103\nOWN - NLM\nSTAT- MEDLINE\nDCOM- 19990225\nLR  - 20190516\nIS  - 0890-9369 (Print)\nIS  - 0890-9369 (Linking)\nVI  - 13\nIP  - 1\nDP  - 1999 Jan 1\nTI  - The Drosophila activin receptor baboon signals through dSmad2 and controls cell\n      proliferation but not patterning during larval development.\nPG  - 98-111\nAB  - The TGF-beta superfamily of growth and differentiation factors, including\n      TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in \n      regulating the development of many organisms. These factors signal through a\n      heteromeric complex of type I and II serine/threonine kinase receptors that\n      phosphorylate members of the Smad family of transcription factors, thereby\n      promoting their nuclear localization. Although components of TGF-beta/Activin\n      signaling pathways are well defined in vertebrates, no such pathway has been\n      clearly defined in invertebrates. In this study we describe the role of Baboon\n      (Babo), a type I Activin receptor previously called Atr-I, in Drosophila\n      development and characterize aspects of the Babo intracellular\n      signal-transduction pathway. Genetic analysis of babo loss-of-function mutants\n      and ectopic activation studies indicate that Babo signaling plays a role in\n      regulating cell proliferation. In mammalian cells, activated Babo specifically\n      stimulates Smad2-dependent pathways to induce TGF-beta/Activin-responsive\n      promoters but not BMP-responsive elements. Furthermore, we identify a new\n      Drosophila Smad, termed dSmad2, that is most closely related to vertebrate Smads \n      2 and 3. Activated Babo associates with dSmad2 but not Mad, phosphorylates the\n      carboxy-terminal SSXS motif and induces heteromeric complex formation with Medea,\n      the Drosophila Smad4 homolog. Our results define a novel Drosophila\n      Activin/TGF-beta pathway that is analogous to its vertebrate counterpart and show\n      that this pathway functions to promote cellular growth with minimal effects on\n      patterning.\nFAU - Brummel, T\nAU  - Brummel T\nAD  - Department of Molecular Biology and Biochemistry, University of California,\n      Irvine, California 92697, USA.\nFAU - Abdollah, S\nAU  - Abdollah S\nFAU - Haerry, T E\nAU  - Haerry TE\nFAU - Shimell, M J\nAU  - Shimell MJ\nFAU - Merriam, J\nAU  - Merriam J\nFAU - Raftery, L\nAU  - Raftery L\nFAU - Wrana, J L\nAU  - Wrana JL\nFAU - O'Connor, M B\nAU  - O'Connor MB\nLA  - eng\nSI  - GENBANK/AF101386\nGR  - GM47462/GM/NIGMS NIH HHS/United States\nPT  - Journal Article\nPT  - Research Support, Non-U.S. Gov't\nPT  - Research Support, U.S. Gov't, P.H.S.\nPL  - United States\nTA  - Genes Dev\nJT  - Genes & development\nJID - 8711660\nRN  - 0 (Bone Morphogenetic Proteins)\nRN  - 0 (DNA-Binding Proteins)\nRN  - 0 (Drosophila Proteins)\nRN  - 0 (RNA, Messenger)\nRN  - 0 (Receptors, Growth Factor)\nRN  - 0 (Smad2 Protein)\nRN  - 0 (Trans-Activators)\nRN  - EC 2.7.11.30 (Activin Receptors)\nRN  - EC 2.7.11.30 (Activin Receptors, Type I)\nRN  - EC 2.7.11.30 (Babo protein, Drosophila)\nSB  - IM\nMH  - Activin Receptors\nMH  - Activin Receptors, Type I\nMH  - Amino Acid Sequence\nMH  - Animals\nMH  - Bone Morphogenetic Proteins/genetics\nMH  - Cell Division\nMH  - Cloning, Molecular\nMH  - DNA-Binding Proteins/chemistry/*genetics\nMH  - Drosophila/*embryology\nMH  - Drosophila Proteins\nMH  - Gene Expression Regulation, Developmental\nMH  - In Situ Hybridization\nMH  - Larva/genetics/*growth & development\nMH  - Molecular Sequence Data\nMH  - Phosphorylation\nMH  - RNA, Messenger/genetics\nMH  - Receptors, Growth Factor/*genetics/metabolism\nMH  - Sequence Alignment\nMH  - Sequence Analysis, DNA\nMH  - Signal Transduction/*physiology\nMH  - Smad2 Protein\nMH  - Trans-Activators/chemistry/*genetics\nMH  - Wings, Animal/growth & development\nPMC - PMC316373\nEDAT- 1999/01/14 00:00\nMHDA- 1999/01/14 00:01\nCRDT- 1999/01/14 00:00\nPHST- 1999/01/14 00:00 [pubmed]\nPHST- 1999/01/14 00:01 [medline]\nPHST- 1999/01/14 00:00 [entrez]\nAID - 10.1101/gad.13.1.98 [doi]\nPST - ppublish\nSO  - Genes Dev. 1999 Jan 1;13(1):98-111. doi: 10.1101/gad.13.1.98.\n"
    #checking if generator is returned, need to check integrity of the returned value
    assert isinstance(_parse_medline(text),types.GeneratorType) == True

def test_uids_to_docs():
    uids = ["9887103"]
    #checking if generator is returned, need to check integrity of the returned value
    assert isinstance(uids_to_docs(uids),types.GeneratorType) == True
