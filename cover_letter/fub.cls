\NeedsTeXFormat{LaTeX2e}[2001/06/01]
\ProvidesClass{fub}[2011/01/01 v1.0
^^JLaTeX Dokumentenklasse fuer den Schriftverkehr der FU Berlin 
^^JCorporate-Design-Handbuch, Bd. 1, S. 21 ff.
^^JUmsetzung: Christoph Eyrich, Berlin]
%---------------------------------------------------
\RequirePackage{ifthen}
   \newboolean{nologo}
\DeclareOption{nologo}{%
   \setboolean{nologo}{true}}
\DeclareOption{brief}{%
   \ifthenelse{\boolean{nologo}}
      {\AtEndOfClass{\geometry{text={149.25mm,228.2mm},top=36.8mm,headsep=5.25mm,ignorehead}}} % 6
      {}
   \AtBeginDocument{%
      \pagestyle{brief}%
      \thispagestyle{brief1stpage}%
      \ifthenelse{\boolean{nologo}}
         {\vspace*{61.5mm}}
         {\vspace*{47mm}}%
      }
   }
\DeclareOption{bericht}{%
   \AtEndOfClass{\geometry{text={149.25mm,228.2mm},top=36.8mm,headsep=4.5mm,ignorehead}}
   \AtBeginDocument{%
      \pagestyle{briefnologo}%
      \thispagestyle{bericht1stpage}%
      \vspace*{41.6mm}%
      }
   }
\DeclareOption{protokoll}{%
   \AtEndOfClass{\geometry{text={149.25mm,228.2mm},top=36.8mm,headsep=4.5mm,ignorehead}}
   \AtBeginDocument{%
      \pagestyle{briefnologo}%
      \thispagestyle{protokoll1stpage}%
      \setlength{\parskip}{3pt}%
      \vspace*{15.45mm}%
      \printprotokollkopf
      }
   \AtEndDocument{%
   \printprotokollschluss
      }
   }
\DeclareOption{vermerk}{%
   \AtEndOfClass{\geometry{text={149.25mm,228.2mm},top=36.8mm,headsep=4.5mm,ignorehead}}
   \AtBeginDocument{%
      \renewcommand{\betreff}[1]{%
         \textbf{#1}\\[1\baselineskip]
         }
      \pagestyle{briefnologo}%
      \thispagestyle{vermerk1stpage}%
      \vspace*{30.45mm}%
      }
   \AtEndDocument{%
      \printvermerkabschluss
      }
   }
\DeclareOption{fax-computer}{%
   \AtBeginDocument{%
      \pagestyle{fax}%
      \thispagestyle{fax-computer1stpage}%
      \fontsize{10.5pt}{14pt}\selectfont
      \vspace*{47mm}%
      }
   }
\DeclareOption{fax-hand}{%
   \AtBeginDocument{%
      \pagestyle{brief}%
      \thispagestyle{fax-hand1stpage}%
      \vspace*{61.28mm}%
      }
   }  
\ProcessOptions
\LoadClass[10pt,a4paper]{article}
\RequirePackage[T1]{fontenc}
\renewcommand{\familydefault}{phv}
\renewcommand\normalsize{%
   \@setfontsize\normalsize\@xpt{13.5pt}%
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalsize
\setlength{\parindent}{0pt}
\setlength{\parskip}{1\baselineskip}
\RequirePackage[a4paper,%
   text={149.25mm,212.75mm},%
   top=51.25mm,left=25mm,%
   head=31.5mm,%
   foot=7mm,%
   headsep=19.75mm,%
   ignorehead]{geometry}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{graphicx}
\RequirePackage{color}
   \definecolor{FUgreen}{rgb}{.73,.797,.199}
\RequirePackage{ragged2e}
   \RaggedRight
\RequirePackage{url}
   \urlstyle{same}
\RequirePackage[flushleft]{paralist}
   \setdefaultenum{1.}{}{}{}
   \setdefaultleftmargin{3em}{}{}{}{}{}
%---
\newcommand{\mintabtl}[1]{%
   \begin{tabular}[t]{@{}l@{}}
      #1
   \end{tabular}%
}
\newcommand{\mintabbl}[1]{%
   \begin{tabular}[b]{@{}l@{}}
      #1
   \end{tabular}%
} 
%--- auszugebende daten ---
\newcommand{\meinfachbereich}{}
\newcommand{\meintitel}{}
\newcommand{\meinvorname}{}
\newcommand{\meinnachname}{}
\newcommand{\meinestrasse}{}
\newcommand{\meinehausnummer}{}
\newcommand{\meinepostleitzahl}{}
\newcommand{\meinetelefonnummer}{}
\newcommand{\meinefaxnummer}{}
\newcommand{\meineemailadresse}{}
\newcommand{\www}{\url{}}
\newcommand{\bearbzeichen}{}
\newcommand{\bearbeiterIn}{}
\newcommand{\ruecksendeadresse}{}
\newcommand{\faxnummer}{}
\newcommand{\faxseitenzahl}{}
\newcommand{\anlagen}[1]{%
   Anlage(n)\\
   \mintabtl{#1}
}
%--- brief ---
\newcommand{\adressvermerk}{}
\newcommand{\empfaengeradresse}{}
\newcommand{\betrefftext}{}
\newcommand{\betreff}[1]{
   \renewcommand{\betrefftext}{#1}%
   \printbetreff
   }
\newcommand{\printbetreff}{%
   \ifthenelse{\equal{\betrefftext}{}}%
      {}%
      {\textbf{\betrefftext}\\[2\baselineskip]}
   }
\newcommand{\anrede}[1]{%
   #1\\[\bigskipamount]
   }
\newcommand{\grussformel}[1]{%
   \mintabtl{#1\\[3\bigskipamount]
   \ifthenelse{\equal{\meintitel}{}}%
      {}%
      {\meintitel\space}%
      \meinvorname\space\meinnachname}
   }
\newlength{\unterschrifthoehe}
\setlength{\unterschrifthoehe}{15mm}
\newcommand{\grussformelsigniert}[1]{%
   \mbox{\rlap{\raisebox{-\unterschrifthoehe}[0pt][0pt]{\includegraphics[height=\unterschrifthoehe]{\unterschriftdatei}}}%
   \mintabtl{#1\\[3\bigskipamount]
   \ifthenelse{\equal{\meintitel}{}}%
      {}%
      {\meintitel\space}%
      \meinvorname\space\meinnachname}}
   }
%--- bericht ---
\newcommand{\titel}[1]{{\fontsize{14pt}{17pt}\selectfont\textbf{#1}}}
\newcommand{\an}{}
%--- protokoll ---
\newcommand{\besprechungsgruppe}{}
\newcommand{\verteiler}{}
\newcommand{\teilnehmendeBesFunktionen}{}
\newcommand{\teilnehmende}{}
\newcommand{\protokollfuehrung}{}
\newcommand{\sitzungsleiterIn}{}
\newcommand{\sitzungsort}{}
\newcommand{\sitzungswochentag}{}
\newcommand{\sitzungsdatum}{}
\newcommand{\sitzungsanfangszeit}{}
\newcommand{\sitzungsschlusszeit}{}
\newcommand{\protokollort}{}
\newcommand{\protokolldatum}{}
\newenvironment{tagesordnung}
   {{\fontsize{10.5pt}{10.5pt}\selectfont\bfseries Tagesordnungspunkte}\medskip
   \setlength{\plitemsep}{10pt}
   \begin{compactenum}
   }
   {\end{compactenum}}
\newenvironment{kommentare}
   {\bigskip
   \setlength{\plitemsep}{10pt}
   \begin{compactenum}}
   {\end{compactenum}}
%--- vermerk ---
\newcommand{\vermerkabsendermitstellenzeichen}{}
\newcommand{\vermerkempfaenger}{}
\newcommand{\vermerkkopiean}{}
\newcommand{\vermerkbearbeiterIn}{}
\newcommand{\appnr}{}
%--- ausgaben ---
\newcommand{\printbriefinfokopf}{%
   \fontencoding{T1}\fontfamily{phv}\fontsize{8.5pt}{11pt}\selectfont
   \begin{tabular}[t]{@{}>{\bfseries\raggedleft\arraybackslash}p{50mm}@{\hspace{2.5mm}}>{\raggedright\arraybackslash}p{55mm}@{}}
      & \smash{\textbf{\mintabtl{\meinfachbereich}}}\\   % feste position der fachbereichsangabe,
      & \mbox{}\\ & \mbox{} \\ & \mbox{} \\              % unabhaengig von deren zeilenzahl
      & \ifthenelse{\equal{\meintitel}{}}{}%
         {\meintitel~}%
        \meinvorname~\meinnachname\\
      & \meinestrasse~\meinehausnummer\\
      & \meinepostleitzahl~Berlin\\[\bigskipamount]
      Telefon        & \meinetelefonnummer\\
%      Fax            & \meinefaxnummer\\
%      Internet       & \www\\
%      Bearb.-Zeichen & \bearbzeichen\\
%      Bearbeiterin   & \bearbeiterIn\\[\bigskipamount]
      E-Mail         & \meineemailadresse\\[\bigskipamount]
      & Berlin, \today
   \end{tabular}%
   }
\newcommand{\printberichtinfokopf}{%
   \fontsize{8.5pt}{11pt}\selectfont
   \mintabtl{%
      \smash{\textbf{\mintabtl{\meinfachbereich}}}\\   % feste position der fachbereichsangabe,
      \mbox{}\\ \mbox{} \\ \mbox{} \\                  % unabhaengig von deren zeilenzahl
      \ifthenelse{\equal{\meintitel}{}}{}%
         {\meintitel\space}%
      \meinvorname\space\meinnachname\\[\bigskipamount]
      \today}
   }   
\newcommand{\printruecksendeadresse}{%
   \fontsize{7pt}{9pt}\selectfont
   \mintabtl{\ruecksendeadresse}%
   }
\newcommand{\printadressvermerk}{%
   \fontsize{9.5pt}{11pt}\selectfont
   \adressvermerk
   }
\newcommand{\printempfaengeradresse}{%
   \fontsize{9.5pt}{11pt}\selectfont
   \mintabtl{\empfaengeradresse}%
   }
\newcommand{\printfaxempfaengerblock}{%
   \begin{tabular}[t]{@{}>{\bfseries}p{20mm}@{}l@{}}
      AN       & \mintabtl{\empfaengeradresse}\\
               & \mbox{}\\
      FAX      & \faxnummer\\[\bigskipamount]
      SEITEN   & (inklusive Deckblatt) \faxseitenzahl
   \end{tabular}
   }
\newcommand{\printfaxadresslinienkopf}{%
   \fontsize{10.5pt}{22pt}\selectfont
   \begin{tabularx}{75mm}[t]{@{}X@{}}
      AN    \hrulefill\\
            \hrulefill\\
      FAX   \hrulefill\\
      VON   \hrulefill\\
            \hrulefill\\
      DATUM \hrulefill\\
      SEITEN (inkl. Deckblatt) \hrulefill
   \end{tabularx}%
   }
\newcommand{\printlinienblock}{%
   \setlength{\unitlength}{1pt}%
   \begin{picture}(0,0)
   \put(0,0){\makebox[\linewidth]{BETREFF \hrulefill}}
   \multiput(0,-22)(0,-22){15}{\rule{\linewidth}{.3pt}}
   \put(0,-352){\makebox[\linewidth]{Mit freundlichen Gr\"u\ss en \hrulefill}}
   \end{picture}
   }
\newcommand{\printprotokollkopf}{%
   {\fontsize{14pt}{17pt}\selectfont\textbf{PROTOKOLL}}\\[2ex]
   \begingroup
   \renewcommand{\arraystretch}{2}%
   \begin{tabularx}{\linewidth}{@{}
      >{\fontsize{10.5pt}{10.5pt}\selectfont\bfseries}p{76mm}
      @{\hspace{1mm}}
      >{\raggedright\arraybackslash}X
         @{}}
   \multicolumn{2}{@{}p{\linewidth}@{}}{%
   \normalfont\fontsize{10.5pt}{10.5pt}\selectfont\besprechungsgruppe} \\ 
   \multicolumn{2}{@{}p{\linewidth}@{}}{%
   \normalfont\verteiler}\\[\bigskipamount]
   \mbox{Teilnehmende mit besonderen Funktionen:} & \teilnehmendeBesFunktionen\\
   Protokollf\"uhrung:       & \protokollfuehrung\\
   Teilnehmende:             & \teilnehmende\\
   Sitzungsort:              & \sitzungsort\\
   Wochentag und Datum:      & \sitzungswochentag\unskip, \sitzungsdatum\\
   Anfangs- und Schlusszeit: & \sitzungsanfangszeit\unskip--\sitzungsschlusszeit\space Uhr
   \end{tabularx}
   \endgroup
   \vspace{1\baselineskip}
   }
\newcommand{\printprotokollschluss}{%
   \vspace{2\baselineskip}
   \fontsize{10pt}{13.5pt}\selectfont
   \begin{tabularx}{\linewidth}{@{}
      p{76mm}
      @{\hspace{1mm}}
      >{\raggedright\arraybackslash}X
         @{}}
      \protokollort     & \protokolldatum\\[2cm]
      \sitzungsleiterIn & \protokollfuehrung\\[\bigskipamount]
      \fontsize{9pt}{11pt}\selectfont\anlagen
   \end{tabularx}
   }   
\newcommand{\printvermerkinfokopf}{%
   \fontsize{10pt}{13.5pt}\selectfont
   \mintabtl{%
      \today\\
      App. \appnr
         }
   }   
\newcommand{\printvermerkabschluss}{%
   \vspace{2\baselineskip}
   \vermerkbearbeiterIn
   }  
%-------------------------------------
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancypagestyle{brief1stpage}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
   \begin{picture}(0,0)
      \put(-25,-75.25){\textcolor{FUgreen}{\rule{10mm}{.3pt}}}
      \put(-25,-118.75){\textcolor{FUgreen}{\rule{10mm}{.3pt}}}
      \put(73,-1.75){\includegraphics[width=76.5mm]{FULogo-RGB}}
      \put(72.5,-9){\printbriefinfokopf}
      \put(0,-21.5){\printruecksendeadresse}
      \put(0,-32){\printadressvermerk}
      \ifthenelse{\equal{\adressvermerk}{}}
         {\put(0,-32){\fontencoding{T1}\fontfamily{phv}\selectfont\printempfaengeradresse}}
         {\put(0,-38){\fontencoding{T1}\fontfamily{phv}\selectfont\printempfaengeradresse}}
    \end{picture}%
      }
}
\fancypagestyle{bericht1stpage}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
   \begin{picture}(0,0)
      \put(-25,-118.75){\textcolor{FUgreen}{\rule{10mm}{.3pt}}}
      \put(73,-1.75){\includegraphics[width=76.5mm]{FULogo-RGB}}
      \put(125,-9){\printberichtinfokopf}
      \put(0,-25){{\fontsize{9.5pt}{11pt}\selectfont An \an}}
    \end{picture}%
        }
}
\fancypagestyle{protokoll1stpage}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}
   \begin{picture}(0,0)
      \put(-25,-118.75){\textcolor{FUgreen}{\rule{10mm}{.3pt}}}
      \put(73,-1.75){\includegraphics[width=76.5mm]{FULogo-RGB}}
    \end{picture}%
        }
}
\fancypagestyle{fax-computer1stpage}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
   \begin{picture}(0,0)
      \put(-25,-118.75){\rule{10mm}{.3pt}}
      \put(73,-1.75){\includegraphics[width=76.5mm]{FULogo-sw}}
      \put(72.5,-9){\printbriefinfokopf}
      \put(0,-26){\printfaxempfaengerblock}  
    \end{picture}%
        }
}
\fancypagestyle{fax-hand1stpage}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
   \begin{picture}(0,0)
      \renewcommand{\today}{}
      \put(-25,-118.75){\rule{10mm}{.3pt}}
      \put(73,-1.8){\includegraphics[width=76.5mm]{FULogo-sw}}
      \put(72.5,-13.4){\printbriefinfokopf}
      \put(0,-21.45){\printfaxadresslinienkopf}
    \end{picture}%
        }
}
\fancypagestyle{vermerk1stpage}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
   \begin{picture}(0,0)
      \put(-25,-118.75){\textcolor{FUgreen}{\rule{10mm}{.3pt}}}
      \put(73,-1.75){\includegraphics[width=76.5mm]{FULogo-RGB}}
      \put(113.5,-9){\printvermerkinfokopf}
      \put(0,-9){\vermerkabsendermitstellenzeichen}
      \put(0,-24.5){\fontsize{14pt}{14pt}\selectfont\textbf{VERMERK}}
      \put(0,-32){%
         \begin{tabularx}{\linewidth}[t]{@{}p{73.5mm}@{\hspace{1mm}}X@{}}
            An  \vermerkempfaenger & Kopie: \vermerkkopiean
         \end{tabularx}
      }
   \end{picture}%
      }
}
\fancypagestyle{brief}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
      \begin{picture}(0,0)
        \put(73,-1.75){\ifthenelse{\boolean{nologo}}
                           {}
                           {\includegraphics[width=76.5mm]{FULogo-RGB}}
                      }
    \end{picture}%
        }
  \lfoot{\thepage}
}
\fancypagestyle{fax}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lhead{%
   \setlength{\unitlength}{1mm}%
      \begin{picture}(0,0)
        \put(73,-1.8){\ifthenelse{\boolean{nologo}}
                           {}
                           {\includegraphics[width=76.5mm]{FULogo-sw}}
                      }
    \end{picture}%
        }
  \lfoot{\thepage}
}
\fancypagestyle{briefnologo}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \lfoot{\thepage}
}
\renewcommand\section{\@startsection{section}{1}{\z@}%
                                     {.8\baselineskip}%
                                     {.2\baselineskip}%
                                     {\normalfont\fontsize{14pt}{16pt}\selectfont}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {.8\baselineskip}%
                                     {.2\baselineskip}%
                                     {\normalfont\normalsize\bfseries\selectfont}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {.8\baselineskip}%
                                     {.2\baselineskip}%
                                     {\normalfont\normalsize\bfseries}}
                                 
