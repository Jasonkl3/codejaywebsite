"use client"
import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeSnippet = ({ codeSnippet }: { codeSnippet: string }) => {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeSnippet);
    } catch (error) {
      console.error(error);
    }
  }
  return (
    <div className="max-w-2xl min-w-[25rem] bg-[#3a404d] rounded-md overflow-hidden">
      <div className="flex justify-between px-4 pt-2 text-white text-xs items-center">
        <p className="text-sm">Python code</p>
        <button onClick={handleCopy} className="py-1 inline-flex items-center gap-1">
          Copy code
        </button>
      </div>
      <SyntaxHighlighter 
        language="python" 
        style={dracula}
        customStyle={{ padding: '25px' }} 
        wrapLongLines={true}
      >
        {codeSnippet}
      </SyntaxHighlighter>
    </div>
  );
};

type Props = {
  codeSnippet: string;
  leetcodeNumber: number;
}

export default function CodeButton(props: Props) {
  const { codeSnippet, leetcodeNumber } = props;
  const [open, setOpen] = useState<boolean>(false);

  const handleClick = () => {
    setOpen(!open);
  }
  
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeSnippet);
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <>
      <button className="font-medium text-blue-600 dark:text-blue-500 hover:underline" onClick={handleClick}>Python</button>
      {open && (
        <div className="modal fixed inset-0 z-50 flex items-center justify-center overflow-y-auto overflow-x-hidden">
          <div className="relative p-4 w-full max-w-2xl max-h-full">
            {/* <!-- Modal content --> */}
            <div className="relative bg-white rounded-lg shadow dark:bg-gray-700">
              <div className="flex items-center justify-between p-4 md:p-5 border-b rounded-t dark:border-gray-600">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Solution</h3>
                <button onClick={handleClick} className="text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 ms-auto inline-flex justify-center items-center dark:hover:bg-gray-600 dark:hover:text-white">
                  <svg className="w-3 h-3" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14">
                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6" />
                  </svg>
                  <span className="sr-only">Close modal</span>
                </button>
              </div>
              {/* <!-- Modal body --> */}
              <div className="overflow-x-scroll p-4 md:p-5 space-y-4">
                <CodeSnippet codeSnippet={codeSnippet} />
              </div>
              {/* <!-- Modal footer --> */}
              <div className="flex items-center p-4 md:p-5 border-t border-gray-200 rounded-b dark:border-gray-600">
                <button onClick={handleCopy} className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">
                  Copy
                </button>
                <button onClick={handleClick} className="py-2.5 px-5 ms-3 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700">
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

{/* <a href={`#code-snippet-${question.leetcodeNumber}`} className="font-medium text-blue-600 dark:text-blue-500 hover:underline">Python</a> */}
// static modal
  // return (
  //   <>
  //     <a href={`#code-snippet-${leetcodeNumber}`} className="font-medium text-blue-600 dark:text-blue-500 hover:underline">Python</a>
  //     <div id={`code-snippet-${leetcodeNumber}`} tabIndex={-1} aria-hidden="true" className="modal hidden overflow-y-auto overflow-x-hidden fixed top-0 right-0 left-0 z-50 justify-center items-center w-full md:inset-0 h-[calc(100%-1rem)] max-h-full">
  //       <div className="relative p-4 w-full max-w-2xl max-h-full">
  //         {/* <!-- Modal content --> */}
  //         <div className="relative bg-white rounded-lg shadow dark:bg-gray-700">
  //           <div className="flex items-center justify-between p-4 md:p-5 border-b rounded-t dark:border-gray-600">
  //             <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Solution</h3>
  //             {/* <a href={`#question-row-${leetcodeNumber}`} className="text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 ms-auto inline-flex justify-center items-center dark:hover:bg-gray-600 dark:hover:text-white"> */}
  //             <button onClick={handleClick} className="text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 ms-auto inline-flex justify-center items-center dark:hover:bg-gray-600 dark:hover:text-white">
  //               <svg className="w-3 h-3" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14">
  //                 <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6" />
  //               </svg>
  //               <span className="sr-only">Close modal</span>
  //             </button>
  //             {/* </a> */}
  //           </div>
  //           {/* <!-- Modal body --> */}
  //           <div className="overflow-x-scroll p-4 md:p-5 space-y-4">
  //             <CodeSnippet codeSnippet={codeSnippet} />
  //           </div>
  //           {/* <!-- Modal footer --> */}
  //           <div className="flex items-center p-4 md:p-5 border-t border-gray-200 rounded-b dark:border-gray-600">
  //             <a href="#" className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">I accept</a>
  //             <a href="#" className="py-2.5 px-5 ms-3 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700">Decline</a>
  //           </div>
  //         </div>
  //       </div>
  //     </div>
  //   </>
  // );
