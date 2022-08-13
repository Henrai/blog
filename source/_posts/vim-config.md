---
title: vim配置 
date: 2022-07-25 18:56:50
tags: vim
top_img: img/vim-cheat-sheet-for-programmers.png
cover: img/vim.png
---

# basic set up:
```vim
set mouse=a     " enable mouse
set nu          " line number   
set tw=4        " tab width
set sw=4 
set et          " convert tab to space
set splitbelow  " new window add below 

filetype on             " set indentation based on the file typpe
filetype plugin on
filetype indent on


syntax on

```

# vim short cut:
ctrl+N: auto complete

# Plugin:

- Airline

- NERDTree:

    - open file in new Tab: `gT`

    - open file : `gt`





# marco
nmap vs nnormap : nmap can recursive

nnoremap <F2> :bp <CR>
nnoremap <F3> :bn <CR>