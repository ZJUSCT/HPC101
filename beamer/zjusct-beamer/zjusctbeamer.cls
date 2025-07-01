\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{zjusct-beamer/zjusctbeamer}[ZJUSCT Beamer Presentation Template]
\LoadClass[aspectratio = 169]{beamer}
\usetheme{metropolis}

% Metropolis global options
% See: http://www.tug.org/texlive//devsrc/Master/texmf-dist/doc/latex/beamertheme-metropolis/metropolistheme.pdf
\metroset{sectionpage = progressbar}
\metroset{subsectionpage = progressbar}
\metroset{progressbar = frametitle}
\metroset{numbering = fraction}
\metroset{block = fill}

% Progress bar thickness
\setlength{\metropolis@titleseparator@linewidth}{2pt}
\setlength{\metropolis@progressonsectionpage@linewidth}{2pt}
\setlength{\metropolis@progressinheadfoot@linewidth}{2pt}

% ZJUSCT colors
\definecolor{silver}{RGB}{127, 127, 132}
\definecolor{blue}{RGB}{0, 82, 155}
\definecolor{zjusct-black}{RGB}{29, 29, 29}
\definecolor{white}{RGB}{255, 255, 255}
\definecolor{darker-black}{RGB}{8, 8, 8}

\setbeamercolor{normal text}{fg = zjusct-black, bg = white}
\setbeamercolor{footer}{fg = white, bg = zjusct-black}
\setbeamercolor{alerted text}{fg = silver, bg = blue}
\setbeamercolor{example text}{fg = blue, bg = white}

% Inter font Settings
\usepackage[sfdefault]{inter}

% Helvetica font Settings
% \usepackage[scaled]{helvet}
% \renewcommand\familydefault{\sfdefault}
% \usepackage[T1]{fontenc}


% Presentation mode
\RequirePackage{pgfpages}
\setbeamertemplate{note page}[plain]

% Replace enumerate from \note[item] with itemize
\def\beamer@setupnote{%
    \gdef\beamer@notesactions{%
        \beamer@outsideframenote{%
            \beamer@atbeginnote%
            \beamer@notes%
            \ifx\beamer@noteitems\@empty\else
                \begin{itemize}\itemsep=0pt\parskip=0pt%
                    \beamer@noteitems%
                \end{itemize}%
            \fi%
            \beamer@atendnote%
        }%
        \gdef\beamer@notesactions{}%
    }
}

% TiZ
\usetikzlibrary{tikzmark}
\usetikzlibrary{shapes, arrows, calc}
\tikzstyle{block} = [rectangle, draw, text width = 40mm, text centered, rounded corners, minimum height = 1.5\baselineskip]
\tikzstyle{line} = [draw, -latex']

% Position blocks anywhere
\RequirePackage[absolute, overlay]{textpos}
\setlength{\TPHorizModule}{\paperwidth}
\setlength{\TPVertModule}{\paperheight}

% Custom blocks
\RequirePackage{tcolorbox}
\tcbuselibrary{skins}
\newtcolorbox{cublock}[1][Title]{
    enhanced,
    colback = white,
    colbacktitle = blue,
    coltitle = white,
    colframe = zjusct-black!70,
    boxrule = 1pt,
    titlerule = 0pt,
    arc = 5pt,
    title = {\strut#1},
    fonttitle = \bfseries,
    fuzzy shadow = {0pt}{-2pt}{0pt}{0.4pt}{black}
}

% Units typesetting
\RequirePackage[binary-units=true]{siunitx}
\sisetup{per-mode=symbol} % fractioned units
\sisetup{range-phrase=\,--\,} % range dash
\sisetup{range-units=single} % range units

% Math
\RequirePackage{mathtools}
\RequirePackage{amssymb}
\RequirePackage{nicefrac}
\RequirePackage{bm}

% Tables
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{adjustbox}
\RequirePackage{longtable}
\RequirePackage{ltablex}
\RequirePackage{multicol}
\RequirePackage{multirow}
\RequirePackage{booktabs}
\RequirePackage{threeparttable}
\RequirePackage{colortbl}
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

% Normal quotes
\RequirePackage{csquotes}
\MakeOuterQuote{"}

% More symbols
\RequirePackage{pifont}

% Captions
\RequirePackage{caption}
\RequirePackage{subcaption}
\captionsetup{labelformat = empty, skip = 10pt, justification = centering}
\captionsetup[subfigure]{labelformat = empty, skip = 10pt, justification = centering}

% Numbered ToC
\setbeamertemplate{section in toc}[sections numbered]
% \setbeamertemplate{subsection in toc}[subsections numbered]

% Extra slides as "appendix"
\RequirePackage{appendixnumberbeamer}
\renewcommand{\appendixname}{\texorpdfstring{\translate{Appendix}}{Appendix}}

% The opposite of '\uncover'
\def\beamer@startmycovered{%
  \def\opaqueness<##1>##2{%
    \only<##1>{%
      \beamer@actions{%
        \expandafter\xdef\csname beamer@oldcolorhook%
        \the\beamer@coveringdepth\endcsname{\beamer@colorhook}%
        \expandafter\xdef\csname beamer@oldpgfextension%
        \the\beamer@coveringdepth\endcsname{\beamer@pgfextension}%
        {\globalcolorstrue\colorlet{beamer@freeze\the\beamer@coveringdepth}{bg}}%
        \xdef\beamer@colorhook{!##2!beamer@freeze%
          \the\beamer@coveringdepth\beamer@colorhook}%
        \gdef\beamer@pgfextension{!##2opaque}%
        \color{.}%
      }%
      {%
        \xdef\beamer@colorhook{\csname beamer@oldcolorhook%
          \the\beamer@coveringdepth\endcsname}%
        \xdef\beamer@pgfextension{\csname beamer@oldpgfextension%
          \the\beamer@coveringdepth\endcsname}%
        \color{.}%
      }}}%
  \ifnum\beamer@slideinframe<\beamer@minimum%ok, at beginning
  {%
    \beamer@saveanother%
    \advance\beamer@minimum by-\beamer@slideinframe%
    \beamer@slideinframe=\beamer@minimum%
    \beamer@uncoverbeforeactions%
    \beamer@restoreanother%
  }%
  \else%
  {%
    \beamer@saveanother%
    \advance\beamer@slideinframe by-\beamer@minimum%
    \beamer@uncoverafteractions%
    \beamer@restoreanother%
  }%
  \fi%
  \beamer@do%
}
\long\def\beamer@makemycovered#1{\beamer@startmycovered#1\beamer@endcovered}
\def\cover{%
\alt{\beamer@makemycovered}{\beamer@fakeinvisible}}
\def\c@slideinframe{\beamer@slideinframe}

% Footer
\setbeamertemplate{footline}{%
    \begin{beamercolorbox}[wd = \textwidth, ht = 0.8cm, dp = 0pt]{footer}
        \hbox to \textwidth{%
            \hspace{0.3cm}%
            \vbox to 0.8cm{\vfill\hbox{\includesvg[height = 0.35cm]{zjusct-beamer/assets/zjusct.svg}}\vfill}%
            \hfill%
            \vbox to 0.8cm{%
                \vfill%
                \hbox{%
                    \insertsubtitle
                    \quad
                    |
                    \quad
                    \insertsection
                    \quad
                    \textbullet
                    \quad
                    \usebeamertemplate{frame numbering}%
                }%
                \vfill%
            }%
            \hspace{0.3cm}%
        }%
    \end{beamercolorbox}%
}

% Keep footer in appendix
\AtBeginDocument{%
    \apptocmd{\appendix}{%
        \setbeamertemplate{footline}{%
        \begin{beamercolorbox}[wd = \textwidth, ht = 1cm, dp = 0pt]{footer}
            \hbox to \textwidth{%
                \hspace{0.3cm}%
                \vbox to 1cm{\vfill\hbox{\includesvg[height = 0.35cm]{zjusct-beamer/assets/zjusct.svg}}\vfill}%
                \hfill%
                \vbox to 1cm{%
                    \vfill%
                    \hbox{%
                        \insertshortauthor
                        \quad
                        \textbullet
                        \quad
                        \insertsubtitle%
                    }%
                    \vfill%
                }%
                \hspace{0.2cm}%
            }%
            % \includegraphics[height = 0.6cm]{img/aems.pdf}
        \end{beamercolorbox}%
    }
    }{}{}
}

% % Use full section name in section pages
% \setbeamertemplate{section page}{
%     \centering
%     \begin{minipage}{22em}
%         \raggedright
%         \usebeamercolor[fg]{section title}
%         \usebeamerfont{section title}
%         \insertsection\\[-1ex]
%         \usebeamertemplate*{progress bar in section page}
%         \par
%         \ifx\insertsubsection\@empty\else%
%             \usebeamercolor[fg]{subsection title}%
%             \usebeamerfont{subsection title}%
%             \insertsubsection
%         \fi
%   \end{minipage}
%   \par
%   \vspace{\baselineskip}
% }

% Add footer to section pages
\AtBeginSection{
    {
        \setbeamertemplate{footline}{
           \begin{beamercolorbox}[wd = \textwidth, ht = 0.8cm, dp = 0pt]{footer}
                \hbox to \textwidth{%
                    \hspace{0.3cm}%
                    \vbox to 0.8cm{\vfill\hbox{\includesvg[height = 0.35cm]{zjusct-beamer/assets/zjusct.svg}}\vfill}%
                    \hfill%
                    \vbox to 0.8cm{%
                        \vfill%
                        \hbox{%
                            \insertshortauthor
                            \quad
                            \textbullet
                            \quad
                            \insertshortsubtitle%
                        }%
                        \vfill%
                    }%
                    \hspace{0.3cm}%
                }%
            \end{beamercolorbox}%
        }
        \frame{
            \sectionpage
        }
        \addtocounter{framenumber}{-1}
    }
}

% Add footer to subsection pages
\AtBeginSubsection{
    {
        \setbeamertemplate{footline}{
           \begin{beamercolorbox}[wd = \textwidth, ht = 0.8cm, dp = 0pt]{footer}
                \hbox to \textwidth{%
                    \hspace{0.3cm}%
                    \vbox to 0.8cm{\vfill\hbox{\includesvg[height = 0.35cm]{zjusct-beamer/assets/zjusct.svg}}\vfill}%
                    \hfill%
                    \vbox to 0.8cm{%
                        \vfill%
                        \hbox{%
                            \insertshortauthor
                            \quad
                            \textbullet
                            \quad
                            \insertshortsubtitle%
                        }%
                        \vfill%
                    }%
                    \hspace{0.3cm}%
                }%
            \end{beamercolorbox}%
        }
        \frame{
            \sectionpage
        }
        \addtocounter{framenumber}{-1}
    }
}

% If one wants to modify the title page... here's the default code
\setbeamertemplate{title page}{%
    \begin{minipage}[b][\paperheight]{\textwidth}%
        \ifx\inserttitlegraphic\@empty\else\usebeamertemplate*{title graphic}\fi%
        \vfill%
        \ifx\inserttitle\@empty\else\usebeamertemplate*{title}\fi%
        \ifx\insertsubtitle\@empty\else\usebeamertemplate*{subtitle}\fi%
        \usebeamertemplate*{title separator}%
        \ifx\beamer@shortauthor\@empty\else\usebeamertemplate*{author}\fi%
        \ifx\insertdate\@empty\else\usebeamertemplate*{date}\fi%
        \ifx\insertinstitute\@empty\else\usebeamertemplate*{institute}\fi%
        \vfill%
        \vspace*{1mm}%
    \end{minipage}%
}

% Add footer to title page
\newcommand*{\copyleftnotice}[2]{%
    \def\@copyleftnotice{#1}%
    \def\@creditnotice{#2}%
}
% Usage:
% 显示完整信息
% \copyleftnotice{CC-BY 4.0}{Slides partially credited to Prof. Wenbo Shen and Prof Rui Chang}
% 只显示版权信息
% \copyleftnotice{CC-BY 4.0}{}
% 只显示 Credit 信息
% \copyleftnotice{}{Slides partially credited to Prof. Wenbo Shen and Prof Rui Chang}
\renewcommand{\maketitle}[1][\item ZJUSCT]{
    \begingroup
        \setbeamercolor{normal text}{fg = white, bg = transparent}
        \setbeamercolor{footer}{fg = white, bg = darker-black}
        \setbeamertemplate{background}{
            \begin{tikzpicture}[remember picture, overlay]
                \fill[top color=zjusct-black, bottom color=darker-black]
                    (current page.south west) rectangle (current page.north east);
            \end{tikzpicture}
        }
        \setbeamertemplate{footline}{
            \begin{beamercolorbox}[wd = \textwidth, ht = 1.2cm, sep = 0.4cm]{footer}
                \includesvg[height = 0.4cm]{zjusct-beamer/assets/zjusct.svg}
                \hfill
                \ifdefined\@copyleftnotice
                    \raisebox{0.22cm}{
                        \begin{minipage}{0.65\paperwidth}
                            \begin{flushright}
                                \ifdefined\@creditnotice
                                    \ifx\@creditnotice\@empty
                                    \else
                                        \@creditnotice
                                        \ifx\@copyleftnotice\@empty\else\\
                                        \fi
                                    \fi
                                \fi
                                \ifx\@copyleftnotice\@empty
                                \else
                                    \@copyleftnotice
                                \fi
                            \end{flushright}
                        \end{minipage}
                    }
                \fi
            \end{beamercolorbox}
        }
        \begin{frame}
            \usebeamertemplate{title page}
            \vspace*{-50pt}
        \end{frame}
        \note[itemize]{#1}
        \addtocounter{framenumber}{-1}
    \endgroup
}

% Table of Contents with "Outline" section name
\newcommand{\cutoc}{
    {
        \metroset{sectionpage = none}
        \section*{Outline}

        \begin{frame}{Outline}
            \tableofcontents[hideallsubsections]
        \end{frame}
    }
}

% Avoid overfull \vbox warnings
\vfuzz = 20pt

% Necessary packages
\RequirePackage{hyperref}
\RequirePackage{listings}
\RequirePackage{minted}
\RequirePackage{paracol}
\RequirePackage{tcolorbox}
\RequirePackage{etoolbox}
\RequirePackage[dvipsnames, svgnames, x11names]{xcolor}

% Emoji & SVG (LuaLaTeX)
\RequirePackage{emoji}
\RequirePackage{svg}
\RequirePackage{indentfirst}
\RequirePackage{luatexja-fontspec}
\svgsetup{inkscapepath=svgsubdir}

% Minted environment
\AtBeginEnvironment{minted}{\let\itshape\relax}
\BeforeBeginEnvironment{minted}%
     {\begin{tcolorbox}}%
\AfterEndEnvironment{minted}
   {\end{tcolorbox}}%

% Minted Style
\setminted{
    fontsize=\smallsize, % 设置代码和行号的字体大小
    breaklines=true, % 自动换行
    style=manni,
    highlightcolor=Khaki1
}

% tcolorbox Style
\tcbset{
    colback=gray!4,  % light gray background
    colframe=white,  % no border
    boxrule=0pt,     % border width is 0
    arc=2mm,         % no rounded corners
    boxsep=2mm,      % small internal padding
    left=1mm,        % left internal padding
    right=1mm,       % right internal padding
    top=1mm,         % top internal padding
    bottom=1mm,      % bottom internal padding
    shadow=true      % add shadow
}

\usetikzlibrary{tikzmark}

\newcounter{mymark}
\newcommand\startunderline{\stepcounter{mymark}\tikzmark{startmark-\themymark}} % bad code from here ...
\newcommand\stopunderline{\tikzmark{stopmark-\themymark}}  % ... up to here
