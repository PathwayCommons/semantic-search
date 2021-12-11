import pytest
import types
from fastapi.exceptions import HTTPException

from semantic_search.ncbi import (
    _medline_to_docs,
    _safe_request,
    _parse_medline,
    _get_eutil_records,
    uids_to_docs,
    Settings,
)

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
    assert _safe_request(url, "POST", files=eutils_params).status_code == 200


def test_get_eutil_records():
    eutil = "efetch"
    _id = "9887103"
    expected = [
        {
            "<!DO": [
                'YPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January '
                '2019//EN" '
                '"https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">'
            ],
            "<?xm": ['version="1.0" ?>'],
            "<Pub": [
                'dArticleSet><PubmedArticle><MedlineCitation Status="MEDLINE" '
                'Owner="NLM"><PMID '
                'Version="1">9887103</PMID><DateCompleted><Year>1999</Year><Month>02</Month><Day>25</Day></DateCompleted><DateRevised><Year>2019</Year><Month>05</Month><Day>16</Day></DateRevised><Article '
                'PubModel="Print"><Journal><ISSN '
                'IssnType="Print">0890-9369</ISSN><JournalIssue '
                'CitedMedium="Print"><Volume>13</Volume><Issue>1</Issue><PubDate><Year>1999</Year><Month>Jan</Month><Day>01</Day></PubDate></JournalIssue><Title>Genes '
                "&amp; development</Title><ISOAbbreviation>Genes "
                "Dev</ISOAbbreviation></Journal><ArticleTitle>The Drosophila activin "
                "receptor baboon signals through dSmad2 and controls cell "
                "proliferation but not patterning during larval "
                "development.</ArticleTitle><Pagination><MedlinePgn>98-111</MedlinePgn></Pagination><Abstract><AbstractText>The "
                "TGF-beta superfamily of growth and differentiation factors, "
                "including TGF-beta, Activins and bone morphogenetic proteins (BMPs) "
                "play critical roles in regulating the development of many "
                "organisms. These factors signal through a heteromeric complex of "
                "type I and II serine/threonine kinase receptors that phosphorylate "
                "members of the Smad family of transcription factors, thereby "
                "promoting their nuclear localization. Although components of "
                "TGF-beta/Activin signaling pathways are well defined in "
                "vertebrates, no such pathway has been clearly defined in "
                "invertebrates. In this study we describe the role of Baboon (Babo), "
                "a type I Activin receptor previously called Atr-I, in Drosophila "
                "development and characterize aspects of the Babo intracellular "
                "signal-transduction pathway. Genetic analysis of babo "
                "loss-of-function mutants and ectopic activation studies indicate "
                "that Babo signaling plays a role in regulating cell proliferation. "
                "In mammalian cells, activated Babo specifically stimulates "
                "Smad2-dependent pathways to induce TGF-beta/Activin-responsive "
                "promoters but not BMP-responsive elements. Furthermore, we identify "
                "a new Drosophila Smad, termed dSmad2, that is most closely related "
                "to vertebrate Smads 2 and 3. Activated Babo associates with dSmad2 "
                "but not Mad, phosphorylates the carboxy-terminal SSXS motif and "
                "induces heteromeric complex formation with Medea, the Drosophila "
                "Smad4 homolog. Our results define a novel Drosophila "
                "Activin/TGF-beta pathway that is analogous to its vertebrate "
                "counterpart and show that this pathway functions to promote "
                "cellular growth with minimal effects on "
                "patterning.</AbstractText></Abstract><AuthorList "
                'CompleteYN="Y"><Author '
                'ValidYN="Y"><LastName>Brummel</LastName><ForeName>T</ForeName><Initials>T</Initials><AffiliationInfo><Affiliation>Department '
                "of Molecular Biology and Biochemistry, University of California, "
                "Irvine, California 92697, "
                "USA.</Affiliation></AffiliationInfo></Author><Author "
                'ValidYN="Y"><LastName>Abdollah</LastName><ForeName>S</ForeName><Initials>S</Initials></Author><Author '
                'ValidYN="Y"><LastName>Haerry</LastName><ForeName>T '
                "E</ForeName><Initials>TE</Initials></Author><Author "
                'ValidYN="Y"><LastName>Shimell</LastName><ForeName>M '
                "J</ForeName><Initials>MJ</Initials></Author><Author "
                'ValidYN="Y"><LastName>Merriam</LastName><ForeName>J</ForeName><Initials>J</Initials></Author><Author '
                'ValidYN="Y"><LastName>Raftery</LastName><ForeName>L</ForeName><Initials>L</Initials></Author><Author '
                'ValidYN="Y"><LastName>Wrana</LastName><ForeName>J '
                "L</ForeName><Initials>JL</Initials></Author><Author "
                'ValidYN="Y"><LastName>O\'Connor</LastName><ForeName>M '
                "B</ForeName><Initials>MB</Initials></Author></AuthorList><Language>eng</Language><DataBankList "
                'CompleteYN="Y"><DataBank><DataBankName>GENBANK</DataBankName><AccessionNumberList><AccessionNumber>AF101386</AccessionNumber></AccessionNumberList></DataBank></DataBankList><GrantList '
                'CompleteYN="Y"><Grant><GrantID>GM47462</GrantID><Acronym>GM</Acronym><Agency>NIGMS '
                "NIH HHS</Agency><Country>United "
                "States</Country></Grant></GrantList><PublicationTypeList><PublicationType "
                'UI="D016428">Journal Article</PublicationType><PublicationType '
                'UI="D013485">Research Support, Non-U.S. '
                'Gov\'t</PublicationType><PublicationType UI="D013487">Research '
                "Support, U.S. Gov't, "
                "P.H.S.</PublicationType></PublicationTypeList></Article><MedlineJournalInfo><Country>United "
                "States</Country><MedlineTA>Genes "
                "Dev</MedlineTA><NlmUniqueID>8711660</NlmUniqueID><ISSNLinking>0890-9369</ISSNLinking></MedlineJournalInfo><ChemicalList><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D019485">Bone Morphogenetic '
                "Proteins</NameOfSubstance></Chemical><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D004268">DNA-Binding '
                "Proteins</NameOfSubstance></Chemical><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D029721">Drosophila '
                "Proteins</NameOfSubstance></Chemical><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D012333">RNA, '
                "Messenger</NameOfSubstance></Chemical><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D017978">Receptors, Growth '
                "Factor</NameOfSubstance></Chemical><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D051899">Smad2 '
                "Protein</NameOfSubstance></Chemical><Chemical><RegistryNumber>0</RegistryNumber><NameOfSubstance "
                'UI="D015534">Trans-Activators</NameOfSubstance></Chemical><Chemical><RegistryNumber>EC '
                '2.7.11.30</RegistryNumber><NameOfSubstance UI="D029404">Activin '
                "Receptors</NameOfSubstance></Chemical><Chemical><RegistryNumber>EC "
                '2.7.11.30</RegistryNumber><NameOfSubstance UI="D030201">Activin '
                "Receptors, Type "
                "I</NameOfSubstance></Chemical><Chemical><RegistryNumber>EC "
                '2.7.11.30</RegistryNumber><NameOfSubstance UI="C517157">Babo '
                "protein, "
                "Drosophila</NameOfSubstance></Chemical></ChemicalList><CitationSubset>IM</CitationSubset><MeshHeadingList><MeshHeading><DescriptorName "
                'UI="D029404" MajorTopicYN="N">Activin '
                "Receptors</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D030201" MajorTopicYN="N">Activin Receptors, Type '
                "I</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D000595" MajorTopicYN="N">Amino Acid '
                "Sequence</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D000818" '
                'MajorTopicYN="N">Animals</DescriptorName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D019485" MajorTopicYN="N">Bone Morphogenetic '
                'Proteins</DescriptorName><QualifierName UI="Q000235" '
                'MajorTopicYN="N">genetics</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D002455" MajorTopicYN="N">Cell '
                "Division</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D003001" MajorTopicYN="N">Cloning, '
                "Molecular</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D004268" MajorTopicYN="N">DNA-Binding '
                'Proteins</DescriptorName><QualifierName UI="Q000737" '
                'MajorTopicYN="N">chemistry</QualifierName><QualifierName '
                'UI="Q000235" '
                'MajorTopicYN="Y">genetics</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D004330" '
                'MajorTopicYN="N">Drosophila</DescriptorName><QualifierName '
                'UI="Q000196" '
                'MajorTopicYN="Y">embryology</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D029721" MajorTopicYN="N">Drosophila '
                "Proteins</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D018507" MajorTopicYN="N">Gene Expression Regulation, '
                "Developmental</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D017403" MajorTopicYN="N">In Situ '
                "Hybridization</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D007814" MajorTopicYN="N">Larva</DescriptorName><QualifierName '
                'UI="Q000235" '
                'MajorTopicYN="N">genetics</QualifierName><QualifierName '
                'UI="Q000254" MajorTopicYN="Y">growth &amp; '
                "development</QualifierName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D008969" MajorTopicYN="N">Molecular Sequence '
                "Data</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D010766" '
                'MajorTopicYN="N">Phosphorylation</DescriptorName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D012333" MajorTopicYN="N">RNA, '
                'Messenger</DescriptorName><QualifierName UI="Q000235" '
                'MajorTopicYN="N">genetics</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D017978" MajorTopicYN="N">Receptors, Growth '
                'Factor</DescriptorName><QualifierName UI="Q000235" '
                'MajorTopicYN="Y">genetics</QualifierName><QualifierName '
                'UI="Q000378" '
                'MajorTopicYN="N">metabolism</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D016415" MajorTopicYN="N">Sequence '
                "Alignment</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D017422" MajorTopicYN="N">Sequence Analysis, '
                "DNA</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D015398" MajorTopicYN="N">Signal '
                'Transduction</DescriptorName><QualifierName UI="Q000502" '
                'MajorTopicYN="Y">physiology</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D051899" MajorTopicYN="N">Smad2 '
                "Protein</DescriptorName></MeshHeading><MeshHeading><DescriptorName "
                'UI="D015534" '
                'MajorTopicYN="N">Trans-Activators</DescriptorName><QualifierName '
                'UI="Q000737" '
                'MajorTopicYN="N">chemistry</QualifierName><QualifierName '
                'UI="Q000235" '
                'MajorTopicYN="Y">genetics</QualifierName></MeshHeading><MeshHeading><DescriptorName '
                'UI="D014921" MajorTopicYN="N">Wings, '
                'Animal</DescriptorName><QualifierName UI="Q000254" '
                'MajorTopicYN="N">growth &amp; '
                "development</QualifierName></MeshHeading></MeshHeadingList></MedlineCitation><PubmedData><History><PubMedPubDate "
                'PubStatus="pubmed"><Year>1999</Year><Month>1</Month><Day>14</Day></PubMedPubDate><PubMedPubDate '
                'PubStatus="medline"><Year>1999</Year><Month>1</Month><Day>14</Day><Hour>0</Hour><Minute>1</Minute></PubMedPubDate><PubMedPubDate '
                'PubStatus="entrez"><Year>1999</Year><Month>1</Month><Day>14</Day><Hour>0</Hour><Minute>0</Minute></PubMedPubDate></History><PublicationStatus>ppublish</PublicationStatus><ArticleIdList><ArticleId '
                'IdType="pubmed">9887103</ArticleId><ArticleId '
                'IdType="pmc">PMC316373</ArticleId><ArticleId '
                'IdType="doi">10.1101/gad.13.1.98</ArticleId></ArticleIdList><ReferenceList><Reference><Citation>Cell. '
                "1992 Dec 11;71(6):1003-14</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">1333888</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell '
                "Tissue Res. 1992 "
                "Jan;267(1):17-28</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">1735111</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1989;107 Suppl:65-74</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">2699859</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1987 Jan 1-7;325(6099):81-4</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">3467201</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1985 Apr;40(4):805-17</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">3986904</ArticleId></ArticleIdList></Reference><Reference><Citation>J '
                "Insect Physiol. 1974 "
                "Jan;20(1):121-41</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">4204331</ArticleId></ArticleIdList></Reference><Reference><Citation>Genetics. '
                "1995 Jan;139(1):241-54</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">7705627</ArticleId></ArticleIdList></Reference><Reference><Citation>J '
                "Biol Chem. 1995 Mar "
                "31;270(13):7134-41</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">7706250</ArticleId></ArticleIdList></Reference><Reference><Citation>Genetics. '
                "1995 Mar;139(3):1347-58</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">7768443</ArticleId></ArticleIdList></Reference><Reference><Citation>EMBO '
                "J. 1995 May 15;14(10):2199-208</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">7774578</ArticleId></ArticleIdList></Reference><Reference><Citation>Genes '
                "Dev. 1994 Nov 1;8(21):2588-601</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">7958918</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1994 Jul 29;78(2):251-61</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8044839</ArticleId></ArticleIdList></Reference><Reference><Citation>Genetika. '
                "1994 Feb;30(2):201-11</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8045382</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1994 Aug 4;370(6488):341-7</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8047140</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1993 Jan;117(1):29-43</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8223253</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1993 Jun;118(2):401-15</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8223268</ArticleId></ArticleIdList></Reference><Reference><Citation>Genetics. '
                "1993 Sep;135(1):71-80</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8224829</ArticleId></ArticleIdList></Reference><Reference><Citation>Mol '
                "Cell Biol. 1994 "
                "Feb;14(2):944-50</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8289834</ArticleId></ArticleIdList></Reference><Reference><Citation>Genes '
                "Dev. 1994 Jan;8(2):133-46</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8299934</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1993 Dec;119(4):1359-69</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8306893</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1993 Apr;117(4):1223-37</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8404527</ArticleId></ArticleIdList></Reference><Reference><Citation>Proc '
                "Natl Acad Sci U S A. 1993 Oct "
                "15;90(20):9475-9</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8415726</ArticleId></ArticleIdList></Reference><Reference><Citation>Proc '
                "Natl Acad Sci U S A. 1993 Apr "
                "1;90(7):2905-9</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8464906</ArticleId></ArticleIdList></Reference><Reference><Citation>Proc '
                "Natl Acad Sci U S A. 1996 Jan "
                "23;93(2):640-5</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8570608</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1996 May 3;85(3):357-68</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8616891</ArticleId></ArticleIdList></Reference><Reference><Citation>Mol '
                "Cell Biol. 1996 "
                "Mar;16(3):1066-73</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8622651</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1996 May;122(5):1555-65</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8625842</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1996 May 30;381(6581):387-93</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8632795</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1996 May 17;85(4):489-500</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8653785</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1996 Jul;122(7):2099-108</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8681791</ArticleId></ArticleIdList></Reference><Reference><Citation>Curr '
                "Opin Genet Dev. 1996 "
                "Aug;6(4):424-31</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8791529</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1996 Oct 24;383(6602):691-6</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8878477</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1996 Oct 31;383(6603):832-6</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8893010</ArticleId></ArticleIdList></Reference><Reference><Citation>EMBO '
                "J. 1996 Dec 2;15(23):6584-94</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8978685</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1996 Dec 27;87(7):1215-24</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">8980228</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1997 Jan;124(1):79-89</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9006069</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1996 Dec;122(12):3939-48</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9012514</ArticleId></ArticleIdList></Reference><Reference><Citation>Curr '
                "Biol. 1997 Apr 1;7(4):270-6</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9094310</ArticleId></ArticleIdList></Reference><Reference><Citation>Genes '
                "Dev. 1997 Apr 15;11(8):984-95</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9136927</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1997 Jun 27;89(7):1165-73</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9215638</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1997 Jul 17;388(6639):304-8</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9230443</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell '
                "Tissue Res. 1997 "
                "Sep;289(3):397-409</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9232819</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1997 Oct 9;389(6651):627-31</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9335506</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1997 Sep;124(18):3555-63</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9342048</ArticleId></ArticleIdList></Reference><Reference><Citation>J '
                "Biol Chem. 1997 Oct "
                "31;272(44):27678-85</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9346908</ArticleId></ArticleIdList></Reference><Reference><Citation>J '
                "Biol Chem. 1997 Oct "
                "31;272(44):28107-15</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9346966</ArticleId></ArticleIdList></Reference><Reference><Citation>Proc '
                "Natl Acad Sci U S A. 1997 Sep "
                "30;94(20):10669-74</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9380693</ArticleId></ArticleIdList></Reference><Reference><Citation>Genes '
                "Dev. 1997 Dec 1;11(23):3157-67</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9389648</ArticleId></ArticleIdList></Reference><Reference><Citation>Nature. '
                "1997 Dec 4;390(6659):465-71</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9393997</ArticleId></ArticleIdList></Reference><Reference><Citation>Biochim '
                "Biophys Acta. 1997 Oct "
                "24;1333(2):F105-50</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9395284</ArticleId></ArticleIdList></Reference><Reference><Citation>Am '
                "J Physiol. 1997 Dec;273(6 Pt "
                "1):L1220-7</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9435577</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 Apr;125(8):1407-20</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9502722</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 Apr;125(8):1433-45</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9502724</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 Apr;125(8):1519-28</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9502733</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 May;125(9):1759-68</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9521913</ArticleId></ArticleIdList></Reference><Reference><Citation>Miner '
                "Electrolyte Metab. "
                "1998;24(2-3):131-5</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9525695</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 May;125(10):1877-87</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9550720</ArticleId></ArticleIdList></Reference><Reference><Citation>Curr '
                "Opin Cell Biol. 1998 "
                "Apr;10(2):188-94</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9561843</ArticleId></ArticleIdList></Reference><Reference><Citation>Biochem '
                "Biophys Res Commun. 1998 May "
                "29;246(3):644-9</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9618266</ArticleId></ArticleIdList></Reference><Reference><Citation>Biochem '
                "Biophys Res Commun. 1998 May "
                "29;246(3):873-80</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9618305</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 Jul;125(14):2723-34</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9636086</ArticleId></ArticleIdList></Reference><Reference><Citation>Cell. '
                "1998 Jun 26;93(7):1183-93</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9657151</ArticleId></ArticleIdList></Reference><Reference><Citation>Genes '
                "Dev. 1998 Jul 15;12(14):2114-9</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9679056</ArticleId></ArticleIdList></Reference><Reference><Citation>Genes '
                "Dev. 1998 Jul 15;12(14):2153-63</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9679060</ArticleId></ArticleIdList></Reference><Reference><Citation>Mol '
                "Cell. 1998 Jul;2(1):109-20</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9702197</ArticleId></ArticleIdList></Reference><Reference><Citation>Development. '
                "1998 Oct;125(20):3977-87</Citation><ArticleIdList><ArticleId "
                'IdType="pubmed">9735359</ArticleId></ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle></PubmedArticleSet>'
            ],
        }
    ]
    actual = _get_eutil_records(eutil, _id)
    assert isinstance(actual, types.GeneratorType)
    assert list(actual) == expected


def test_parse_medline():
    text = "\nPMID- 9887103\nOWN - NLM\nSTAT- MEDLINE\nDCOM- 19990225\nLR  - 20190516\nIS  - 0890-9369 (Print)\nIS  - 0890-9369 (Linking)\nVI  - 13\nIP  - 1\nDP  - 1999 Jan 1\nTI  - The Drosophila activin receptor baboon signals through dSmad2 and controls cell\n      proliferation but not patterning during larval development.\nPG  - 98-111\nAB  - The TGF-beta superfamily of growth and differentiation factors, including\n      TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in \n      regulating the development of many organisms. These factors signal through a\n      heteromeric complex of type I and II serine/threonine kinase receptors that\n      phosphorylate members of the Smad family of transcription factors, thereby\n      promoting their nuclear localization. Although components of TGF-beta/Activin\n      signaling pathways are well defined in vertebrates, no such pathway has been\n      clearly defined in invertebrates. In this study we describe the role of Baboon\n      (Babo), a type I Activin receptor previously called Atr-I, in Drosophila\n      development and characterize aspects of the Babo intracellular\n      signal-transduction pathway. Genetic analysis of babo loss-of-function mutants\n      and ectopic activation studies indicate that Babo signaling plays a role in\n      regulating cell proliferation. In mammalian cells, activated Babo specifically\n      stimulates Smad2-dependent pathways to induce TGF-beta/Activin-responsive\n      promoters but not BMP-responsive elements. Furthermore, we identify a new\n      Drosophila Smad, termed dSmad2, that is most closely related to vertebrate Smads \n      2 and 3. Activated Babo associates with dSmad2 but not Mad, phosphorylates the\n      carboxy-terminal SSXS motif and induces heteromeric complex formation with Medea,\n      the Drosophila Smad4 homolog. Our results define a novel Drosophila\n      Activin/TGF-beta pathway that is analogous to its vertebrate counterpart and show\n      that this pathway functions to promote cellular growth with minimal effects on\n      patterning.\nFAU - Brummel, T\nAU  - Brummel T\nAD  - Department of Molecular Biology and Biochemistry, University of California,\n      Irvine, California 92697, USA.\nFAU - Abdollah, S\nAU  - Abdollah S\nFAU - Haerry, T E\nAU  - Haerry TE\nFAU - Shimell, M J\nAU  - Shimell MJ\nFAU - Merriam, J\nAU  - Merriam J\nFAU - Raftery, L\nAU  - Raftery L\nFAU - Wrana, J L\nAU  - Wrana JL\nFAU - O'Connor, M B\nAU  - O'Connor MB\nLA  - eng\nSI  - GENBANK/AF101386\nGR  - GM47462/GM/NIGMS NIH HHS/United States\nPT  - Journal Article\nPT  - Research Support, Non-U.S. Gov't\nPT  - Research Support, U.S. Gov't, P.H.S.\nPL  - United States\nTA  - Genes Dev\nJT  - Genes & development\nJID - 8711660\nRN  - 0 (Bone Morphogenetic Proteins)\nRN  - 0 (DNA-Binding Proteins)\nRN  - 0 (Drosophila Proteins)\nRN  - 0 (RNA, Messenger)\nRN  - 0 (Receptors, Growth Factor)\nRN  - 0 (Smad2 Protein)\nRN  - 0 (Trans-Activators)\nRN  - EC 2.7.11.30 (Activin Receptors)\nRN  - EC 2.7.11.30 (Activin Receptors, Type I)\nRN  - EC 2.7.11.30 (Babo protein, Drosophila)\nSB  - IM\nMH  - Activin Receptors\nMH  - Activin Receptors, Type I\nMH  - Amino Acid Sequence\nMH  - Animals\nMH  - Bone Morphogenetic Proteins/genetics\nMH  - Cell Division\nMH  - Cloning, Molecular\nMH  - DNA-Binding Proteins/chemistry/*genetics\nMH  - Drosophila/*embryology\nMH  - Drosophila Proteins\nMH  - Gene Expression Regulation, Developmental\nMH  - In Situ Hybridization\nMH  - Larva/genetics/*growth & development\nMH  - Molecular Sequence Data\nMH  - Phosphorylation\nMH  - RNA, Messenger/genetics\nMH  - Receptors, Growth Factor/*genetics/metabolism\nMH  - Sequence Alignment\nMH  - Sequence Analysis, DNA\nMH  - Signal Transduction/*physiology\nMH  - Smad2 Protein\nMH  - Trans-Activators/chemistry/*genetics\nMH  - Wings, Animal/growth & development\nPMC - PMC316373\nEDAT- 1999/01/14 00:00\nMHDA- 1999/01/14 00:01\nCRDT- 1999/01/14 00:00\nPHST- 1999/01/14 00:00 [pubmed]\nPHST- 1999/01/14 00:01 [medline]\nPHST- 1999/01/14 00:00 [entrez]\nAID - 10.1101/gad.13.1.98 [doi]\nPST - ppublish\nSO  - Genes Dev. 1999 Jan 1;13(1):98-111. doi: 10.1101/gad.13.1.98.\n"
    expected = [
        {
            "PMID": "9887103",
            "OWN": "NLM",
            "STAT": "MEDLINE",
            "DCOM": "19990225",
            "LR": "20190516",
            "IS": "0890-9369 (Print) 0890-9369 (Linking)",
            "VI": "13",
            "IP": "1",
            "DP": "1999 Jan 1",
            "TI": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development.",
            "PG": "98-111",
            "AB": "The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms. These factors signal through a heteromeric complex of type I and II serine/threonine kinase receptors that phosphorylate members of the Smad family of transcription factors, thereby promoting their nuclear localization. Although components of TGF-beta/Activin signaling pathways are well defined in vertebrates, no such pathway has been clearly defined in invertebrates. In this study we describe the role of Baboon (Babo), a type I Activin receptor previously called Atr-I, in Drosophila development and characterize aspects of the Babo intracellular signal-transduction pathway. Genetic analysis of babo loss-of-function mutants and ectopic activation studies indicate that Babo signaling plays a role in regulating cell proliferation. In mammalian cells, activated Babo specifically stimulates Smad2-dependent pathways to induce TGF-beta/Activin-responsive promoters but not BMP-responsive elements. Furthermore, we identify a new Drosophila Smad, termed dSmad2, that is most closely related to vertebrate Smads 2 and 3. Activated Babo associates with dSmad2 but not Mad, phosphorylates the carboxy-terminal SSXS motif and induces heteromeric complex formation with Medea, the Drosophila Smad4 homolog. Our results define a novel Drosophila Activin/TGF-beta pathway that is analogous to its vertebrate counterpart and show that this pathway functions to promote cellular growth with minimal effects on patterning.",
            "FAU": [
                "Brummel, T",
                "Abdollah, S",
                "Haerry, T E",
                "Shimell, M J",
                "Merriam, J",
                "Raftery, L",
                "Wrana, J L",
                "O'Connor, M B",
            ],
            "AU": [
                "Brummel T",
                "Abdollah S",
                "Haerry TE",
                "Shimell MJ",
                "Merriam J",
                "Raftery L",
                "Wrana JL",
                "O'Connor MB",
            ],
            "AD": [
                "Department of Molecular Biology and Biochemistry, University of California, Irvine, California 92697, USA."
            ],
            "LA": ["eng"],
            "SI": ["GENBANK/AF101386"],
            "GR": ["GM47462/GM/NIGMS NIH HHS/United States"],
            "PT": [
                "Journal Article",
                "Research Support, Non-U.S. Gov't",
                "Research Support, U.S. Gov't, P.H.S.",
            ],
            "PL": "United States",
            "TA": "Genes Dev",
            "JT": "Genes & development",
            "JID": "8711660",
            "RN": [
                "0 (Bone Morphogenetic Proteins)",
                "0 (DNA-Binding Proteins)",
                "0 (Drosophila Proteins)",
                "0 (RNA, Messenger)",
                "0 (Receptors, Growth Factor)",
                "0 (Smad2 Protein)",
                "0 (Trans-Activators)",
                "EC 2.7.11.30 (Activin Receptors)",
                "EC 2.7.11.30 (Activin Receptors, Type I)",
                "EC 2.7.11.30 (Babo protein, Drosophila)",
            ],
            "SB": "IM",
            "MH": [
                "Activin Receptors",
                "Activin Receptors, Type I",
                "Amino Acid Sequence",
                "Animals",
                "Bone Morphogenetic Proteins/genetics",
                "Cell Division",
                "Cloning, Molecular",
                "DNA-Binding Proteins/chemistry/*genetics",
                "Drosophila/*embryology",
                "Drosophila Proteins",
                "Gene Expression Regulation, Developmental",
                "In Situ Hybridization",
                "Larva/genetics/*growth & development",
                "Molecular Sequence Data",
                "Phosphorylation",
                "RNA, Messenger/genetics",
                "Receptors, Growth Factor/*genetics/metabolism",
                "Sequence Alignment",
                "Sequence Analysis, DNA",
                "Signal Transduction/*physiology",
                "Smad2 Protein",
                "Trans-Activators/chemistry/*genetics",
                "Wings, Animal/growth & development",
            ],
            "PMC": "PMC316373",
            "EDAT": "1999/01/14 00:00",
            "MHDA": "1999/01/14 00:01",
            "CRDT": ["1999/01/14 00:00"],
            "PHST": [
                "1999/01/14 00:00 [pubmed]",
                "1999/01/14 00:01 [medline]",
                "1999/01/14 00:00 [entrez]",
            ],
            "AID": ["10.1101/gad.13.1.98 [doi]"],
            "PST": "ppublish",
            "SO": "Genes Dev. 1999 Jan 1;13(1):98-111. doi: 10.1101/gad.13.1.98.",
        }
    ]
    actual = _parse_medline(text)
    assert isinstance(actual, types.GeneratorType)
    assert list(actual) == expected


def test_uids_to_docs():
    uids = ["9887103"]
    expected = [
        [
            {
                "uid": "9887103",
                "text": "The Drosophila activin receptor baboon signals through dSmad2 and controls cell proliferation but not patterning during larval development. The TGF-beta superfamily of growth and differentiation factors, including TGF-beta, Activins and bone morphogenetic proteins (BMPs) play critical roles in regulating the development of many organisms. These factors signal through a heteromeric complex of type I and II serine/threonine kinase receptors that phosphorylate members of the Smad family of transcription factors, thereby promoting their nuclear localization. Although components of TGF-beta/Activin signaling pathways are well defined in vertebrates, no such pathway has been clearly defined in invertebrates. In this study we describe the role of Baboon (Babo), a type I Activin receptor previously called Atr-I, in Drosophila development and characterize aspects of the Babo intracellular signal-transduction pathway. Genetic analysis of babo loss-of-function mutants and ectopic activation studies indicate that Babo signaling plays a role in regulating cell proliferation. In mammalian cells, activated Babo specifically stimulates Smad2-dependent pathways to induce TGF-beta/Activin-responsive promoters but not BMP-responsive elements. Furthermore, we identify a new Drosophila Smad, termed dSmad2, that is most closely related to vertebrate Smads 2 and 3. Activated Babo associates with dSmad2 but not Mad, phosphorylates the carboxy-terminal SSXS motif and induces heteromeric complex formation with Medea, the Drosophila Smad4 homolog. Our results define a novel Drosophila Activin/TGF-beta pathway that is analogous to its vertebrate counterpart and show that this pathway functions to promote cellular growth with minimal effects on patterning.",
            }
        ]
    ]
    actual = uids_to_docs(uids)
    assert isinstance(actual, types.GeneratorType)
    assert list(actual) == expected
